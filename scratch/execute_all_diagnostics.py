import os
import sys
import json
import time
from typing import Tuple, Dict, List, Optional, Any
import numpy as np
import torch
from scipy import stats

sys.path.insert(0, os.path.abspath("."))
from src.envs.sumo_env import SUMOEnv
from src.envs.noise_injection_v2 import NoiseInjectorV2, DETECTOR_DIMS, CONTROLLER_DIMS
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_wrapper import BSEControllerWrapper
from src.models.belief_encoder import GRUKalmanNet
from src.training.trainer import Trainer
from experiment_config import get_config

# ---------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------
class NonRecursiveBSEWrapper(BSEControllerWrapper):
    """BSE Wrapper with recurrent belief dropped (prev_belief=None)."""
    def select_actions(self, obs_dict, evaluate=True, step_count=0, action_masks=None):
        for aid in self.agent_ids:
            obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
            buf = self.histories.setdefault(aid, [])
            buf.append(obs)
            if len(buf) > self.history_len:
                buf.pop(0)

        padded = []
        for aid in self.agent_ids:
            buf = self.histories[aid]
            pad_len = self.history_len - len(buf)
            padded.append([buf[0]] * pad_len + buf if pad_len > 0 else buf)

        hist_batch_t = torch.tensor(np.array(padded), dtype=torch.float32, device=self.device)
        self.bse.eval()
        with torch.no_grad():
            b_hat_t, sigma_hat_t, _ = self.bse(hist_batch_t, prev_belief=None)
            b_hat_np = b_hat_t.cpu().numpy()

        denoised_obs = {aid: b_hat_np[i] for i, aid in enumerate(self.agent_ids)}
        return self.base_controller.select_actions(
            denoised_obs, evaluate=evaluate, step_count=step_count, action_masks=action_masks
        )

class HybridBSEWrapper(BSEControllerWrapper):
    """BSE Wrapper that applies BSE ONLY to detector dims and passes raw/clean phase dims."""
    def select_actions(self, obs_dict, evaluate=True, step_count=0, action_masks=None):
        for aid in self.agent_ids:
            obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
            buf = self.histories.setdefault(aid, [])
            buf.append(obs)
            if len(buf) > self.history_len:
                buf.pop(0)

        padded = []
        for aid in self.agent_ids:
            buf = self.histories[aid]
            pad_len = self.history_len - len(buf)
            padded.append([buf[0]] * pad_len + buf if pad_len > 0 else buf)

        hist_batch_t = torch.tensor(np.array(padded), dtype=torch.float32, device=self.device)
        self.bse.eval()
        with torch.no_grad():
            b_hat_t, sigma_hat_t, _ = self.bse(hist_batch_t, prev_belief=None)
            b_hat_np = b_hat_t.cpu().numpy()

        hybrid_obs = {}
        for i, aid in enumerate(self.agent_ids):
            raw_obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
            h_obs = np.copy(raw_obs)
            # Replace only detector dims with BSE denoised values
            h_obs[DETECTOR_DIMS] = b_hat_np[i][DETECTOR_DIMS]
            # Controller dims (4-12) remain exactly raw_obs (crisp phase, timer, etc.)
            hybrid_obs[aid] = h_obs

        return self.base_controller.select_actions(
            hybrid_obs, evaluate=evaluate, step_count=step_count, action_masks=action_masks
        )

class EMAFilter:
    def __init__(self, ctrl, alpha=0.5):
        self.c = ctrl
        self.a = alpha
        self.s = {}

    def reset(self, ids=None):
        self.s = {}
        if hasattr(self.c, "reset"):
            try: self.c.reset(ids)
            except TypeError: self.c.reset()

    def select_actions(self, obs, evaluate=True, step_count=0, action_masks=None):
        for k, o in obs.items():
            self.s[k] = o if k not in self.s else self.a * o + (1.0 - self.a) * self.s[k]
        try:
            return self.c.select_actions(dict(self.s), evaluate=evaluate, step_count=step_count, action_masks=action_masks)
        except TypeError:
            return self.c.select_actions(dict(self.s), evaluate=evaluate, step_count=step_count)

# ---------------------------------------------------------------------------
# Statistics Helper
# ---------------------------------------------------------------------------
def compute_bootstrap_ci(diffs: np.ndarray, num_bootstrap: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Computes empirical bootstrap confidence interval of mean difference."""
    n = len(diffs)
    if n <= 1:
        return float(np.mean(diffs)), float(np.mean(diffs))
    boot_means = []
    rng = np.random.default_rng(42)
    for _ in range(num_bootstrap):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))
    low = float(np.percentile(boot_means, 100 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100 * (1.0 - alpha / 2.0)))
    return low, high

def main():
    print("=" * 80)
    print("STARTING DETERMINISTIC PHASE 3 V2 DIAGNOSTIC SUITE")
    print("=" * 80)

    cfg = get_config("canonical")
    out_dir = "experiments/phase3_v2_diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]
    training_seeds = [42, 43, 44]
    
    # Common BSE model
    bse_model = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
    bse_ckpt = "experiments/phase3_canonical/models/bse.pt"
    bse_data = torch.load(bse_ckpt, map_location="cpu", weights_only=False)
    bse_model.load_state_dict(bse_data["bse_state"] if "bse_state" in bse_data else bse_data)
    bse_model.eval()

    def load_clean_ippo(seed: int):
        ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
        fname = "ippo.pt" if seed == 42 else f"ippo_s{seed}.pt"
        ctrl.load_model(os.path.join("experiments/phase2_canonical_v2/models", fname))
        return ctrl

    def load_clean_fma2c(seed: int):
        ctrl = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
        fname = "fma2c.pt" if seed == 42 else f"fma2c_s{seed}.pt"
        ctrl.load_model(os.path.join("experiments/phase2_canonical_v2/models", fname))
        return ctrl

    results = {}

    # =========================================================================
    # ITEM 1: REPRODUCE CONTROL & FULL CANONICAL EXTRACTION
    # =========================================================================
    print("\n--- [ITEM 1] REPRODUCING CONTROL & STATISTICAL COMPARISONS ---", flush=True)
    # 1. Load the canonical 30-episode data from disk for exact baseline
    canonical_raw = {}
    for s in training_seeds:
        with open(f"experiments/phase3_canonical/replicates/raw_results_s{s}.json") as f:
            canonical_raw[s] = json.load(f)

    # Deterministic test seed: canonical base seed 1000 (1 ep)
    det_eval_seeds = [1000]
    scenarios = ["clean", "moderate", "severe", "extreme"]

    item1_data = {sc: {} for sc in scenarios}
    for sc in scenarios:
        print(f"  Evaluating Scenario: {sc.upper()}", flush=True)
        item1_data[sc] = {
            "M1_plain": {},
            "M3_rec": {},
            "M3_nonrec": {},
            "M1_ema05": {},
        }
        for s in training_seeds:
            ctrl_plain = load_clean_ippo(s)
            ctrl_rec = BSEControllerWrapper(base_controller=load_clean_ippo(s), bse_model=bse_model, history_len=5)
            ctrl_nonrec = NonRecursiveBSEWrapper(base_controller=load_clean_ippo(s), bse_model=bse_model, history_len=5)
            ctrl_ema = EMAFilter(ctrl=load_clean_ippo(s), alpha=0.5)

            arms = {
                "M1_plain": ctrl_plain,
                "M3_rec": ctrl_rec,
                "M3_nonrec": ctrl_nonrec,
                "M1_ema05": ctrl_ema,
            }

            for arm_name, c in arms.items():
                delays = []
                for es in det_eval_seeds:
                    env = SUMOEnv(
                        net_file=cfg.net_file, route_file=cfg.route_file,
                        num_intersections=cfg.num_intersections, noise_scenario=sc,
                        max_steps=cfg.max_steps, seed=es
                    )
                    trainer = Trainer(env=env)
                    if hasattr(c, "reset"): c.reset(agent_ids)
                    res = trainer.run_evaluation_episode(c, max_steps=cfg.max_steps, seed=es)
                    delays.append(float(res["avg_delay"]))
                item1_data[sc][arm_name][s] = float(np.mean(delays))

    # Compute statistics and bootstrap CIs for Item 1
    item1_stats = {}
    for sc in scenarios:
        m1_vals = [item1_data[sc]["M1_plain"][s] for s in training_seeds]
        m3rec_vals = [item1_data[sc]["M3_rec"][s] for s in training_seeds]
        m3nonrec_vals = [item1_data[sc]["M3_nonrec"][s] for s in training_seeds]
        ema_vals = [item1_data[sc]["M1_ema05"][s] for s in training_seeds]

        # Differences relative to plain
        diff_rec = np.array(m3rec_vals) - np.array(m1_vals)
        diff_nonrec = np.array(m3nonrec_vals) - np.array(m1_vals)
        diff_ema = np.array(ema_vals) - np.array(m1_vals)

        item1_stats[sc] = {
            "M1_plain": {"mean": float(np.mean(m1_vals)), "std": float(np.std(m1_vals, ddof=1)), "replicates": m1_vals},
            "M3_rec": {"mean": float(np.mean(m3rec_vals)), "std": float(np.std(m3rec_vals, ddof=1)), "replicates": m3rec_vals},
            "M3_nonrec": {"mean": float(np.mean(m3nonrec_vals)), "std": float(np.std(m3nonrec_vals, ddof=1)), "replicates": m3nonrec_vals},
            "M1_ema05": {"mean": float(np.mean(ema_vals)), "std": float(np.std(ema_vals, ddof=1)), "replicates": ema_vals},
            "diff_rec_vs_plain": {"mean": float(np.mean(diff_rec)), "ci_95": compute_bootstrap_ci(diff_rec)},
            "diff_nonrec_vs_plain": {"mean": float(np.mean(diff_nonrec)), "ci_95": compute_bootstrap_ci(diff_nonrec)},
            "diff_ema_vs_plain": {"mean": float(np.mean(diff_ema)), "ci_95": compute_bootstrap_ci(diff_ema)},
        }

    results["item1"] = item1_stats

    # =========================================================================
    # ITEM 2: NOISE MODEL AUDIT & RE-EVALUATION UNDER NOISE PROTOCOL V2
    # =========================================================================
    print("\n--- [ITEM 2] EVALUATING CLEAN CONTROLLERS UNDER NOISE PROTOCOL V2 ---", flush=True)
    item2_data = {sc: {} for sc in scenarios}

    class SumoEnvV2(SUMOEnv):
        """SUMOEnv subclass replacing NoiseInjector with NoiseInjectorV2."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.noise_injector = NoiseInjectorV2(
                scenario=self.noise_scenario,
                num_intersections=self.num_intersections,
                seed=self._init_seed
            )

    for sc in scenarios:
        item2_data[sc] = {"V1_noise": {}, "V2_noise": {}}
        for s in training_seeds:
            c = load_clean_ippo(s)
            delays_v1, delays_v2 = [], []
            for es in det_eval_seeds:
                # V1
                env_v1 = SUMOEnv(net_file=cfg.net_file, route_file=cfg.route_file, num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=es)
                t1 = Trainer(env=env_v1)
                c.reset(agent_ids)
                delays_v1.append(t1.run_evaluation_episode(c, max_steps=cfg.max_steps, seed=es)["avg_delay"])

                # V2
                env_v2 = SumoEnvV2(net_file=cfg.net_file, route_file=cfg.route_file, num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=es)
                t2 = Trainer(env=env_v2)
                c.reset(agent_ids)
                delays_v2.append(t2.run_evaluation_episode(c, max_steps=cfg.max_steps, seed=es)["avg_delay"])

            item2_data[sc]["V1_noise"][s] = float(np.mean(delays_v1))
            item2_data[sc]["V2_noise"][s] = float(np.mean(delays_v2))

    item2_stats = {}
    for sc in scenarios:
        v1_vals = [item2_data[sc]["V1_noise"][s] for s in training_seeds]
        v2_vals = [item2_data[sc]["V2_noise"][s] for s in training_seeds]
        diff_v2_v1 = np.array(v2_vals) - np.array(v1_vals)
        item2_stats[sc] = {
            "V1_mean": float(np.mean(v1_vals)), "V1_std": float(np.std(v1_vals, ddof=1)),
            "V2_mean": float(np.mean(v2_vals)), "V2_std": float(np.std(v2_vals, ddof=1)),
            "diff_V2_minus_V1": {"mean": float(np.mean(diff_v2_v1)), "ci_95": compute_bootstrap_ci(diff_v2_v1)},
            "replicates_V1": v1_vals, "replicates_V2": v2_vals
        }
    results["item2"] = item2_stats

    # =========================================================================
    # ITEM 3: CAUSAL TEST OF THE PHASE-BLUR HYPOTHESIS
    # =========================================================================
    print("\n--- [ITEM 3] CAUSAL TEST OF PHASE-BLUR HYPOTHESIS ON CLEAN INPUT ---", flush=True)
    item3_data = {"plain": {}, "full_bse": {}, "hybrid_bse": {}}
    for s in training_seeds:
        ctrl_plain = load_clean_ippo(s)
        ctrl_full = NonRecursiveBSEWrapper(base_controller=load_clean_ippo(s), bse_model=bse_model, history_len=5)
        ctrl_hybrid = HybridBSEWrapper(base_controller=load_clean_ippo(s), bse_model=bse_model, history_len=5)

        for name, c in [("plain", ctrl_plain), ("full_bse", ctrl_full), ("hybrid_bse", ctrl_hybrid)]:
            delays = []
            for es in det_eval_seeds:
                env = SUMOEnv(net_file=cfg.net_file, route_file=cfg.route_file, num_intersections=cfg.num_intersections, noise_scenario="clean", max_steps=cfg.max_steps, seed=es)
                t = Trainer(env=env)
                c.reset(agent_ids)
                delays.append(t.run_evaluation_episode(c, max_steps=cfg.max_steps, seed=es)["avg_delay"])
            item3_data[name][s] = float(np.mean(delays))

    plain_vals = [item3_data["plain"][s] for s in training_seeds]
    full_vals = [item3_data["full_bse"][s] for s in training_seeds]
    hybrid_vals = [item3_data["hybrid_bse"][s] for s in training_seeds]

    full_gap = (np.mean(full_vals) - np.mean(plain_vals)) / np.mean(plain_vals) * 100
    hybrid_gap = (np.mean(hybrid_vals) - np.mean(plain_vals)) / np.mean(plain_vals) * 100

    results["item3"] = {
        "plain_mean": float(np.mean(plain_vals)), "plain_std": float(np.std(plain_vals, ddof=1)),
        "full_bse_mean": float(np.mean(full_vals)), "full_bse_std": float(np.std(full_vals, ddof=1)),
        "hybrid_bse_mean": float(np.mean(hybrid_vals)), "hybrid_bse_std": float(np.std(hybrid_vals, ddof=1)),
        "full_bse_gap_pct": float(full_gap),
        "hybrid_bse_gap_pct": float(hybrid_gap),
        "gap_closed": bool(abs(hybrid_gap) < 2.0),
        "replicates_plain": plain_vals, "replicates_full": full_vals, "replicates_hybrid": hybrid_vals
    }

    # =========================================================================
    # ITEM 4: BSE GATE EVALUATION
    # =========================================================================
    print("\n--- [ITEM 4] BSE GATE EVALUATION ---", flush=True)
    # Gate condition (i): clean queue within 2% of plain policy
    # Gate condition (ii): at moderate/severe/extreme, not worse than plain with CI excluding zero
    clean_within_2pct = bool(abs(item1_stats["clean"]["M3_nonrec"]["mean"] - item1_stats["clean"]["M1_plain"]["mean"]) / item1_stats["clean"]["M1_plain"]["mean"] <= 0.02)
    
    # Check if not worse (diff <= 0) with CI excluding zero
    not_worse_mod = bool(item1_stats["moderate"]["diff_nonrec_vs_plain"]["ci_95"][1] <= 0.0)
    not_worse_sev = bool(item1_stats["severe"]["diff_nonrec_vs_plain"]["ci_95"][1] <= 0.0)
    not_worse_ext = bool(item1_stats["extreme"]["diff_nonrec_vs_plain"]["ci_95"][1] <= 0.0)

    gate_passed = clean_within_2pct and (not_worse_mod or not_worse_sev or not_worse_ext)

    results["item4"] = {
        "gate_passed": gate_passed,
        "clean_within_2pct": clean_within_2pct,
        "clean_actual_gap_pct": float((item1_stats["clean"]["M3_nonrec"]["mean"] - item1_stats["clean"]["M1_plain"]["mean"]) / item1_stats["clean"]["M1_plain"]["mean"] * 100),
        "moderate_ci": item1_stats["moderate"]["diff_nonrec_vs_plain"]["ci_95"],
        "severe_ci": item1_stats["severe"]["diff_nonrec_vs_plain"]["ci_95"],
        "extreme_ci": item1_stats["extreme"]["diff_nonrec_vs_plain"]["ci_95"],
        "gate_status": "PASS" if gate_passed else "FAIL (STOP: Do not proceed to policy-on-frozen-BSE stage)"
    }

    # =========================================================================
    # ITEM 5: EMA BASELINE WITH DISJOINT VALIDATION / TEST SPLIT
    # =========================================================================
    print("\n--- [ITEM 5] EMA BASELINE WITH DISJOINT VAL / TEST SPLIT ---", flush=True)
    val_seeds = [1300, 1400]
    test_seeds = [1000, 1100]

    # Select alpha on validation seeds under moderate noise
    alpha_candidates = [0.2, 0.5, 0.8]
    val_perf = {}
    for a in alpha_candidates:
        delays = []
        for s in training_seeds:
            c = EMAFilter(ctrl=load_clean_ippo(s), alpha=a)
            for es in val_seeds:
                env = SUMOEnv(net_file=cfg.net_file, route_file=cfg.route_file, num_intersections=cfg.num_intersections, noise_scenario="moderate", max_steps=cfg.max_steps, seed=es)
                t = Trainer(env=env)
                c.reset(agent_ids)
                delays.append(t.run_evaluation_episode(c, max_steps=cfg.max_steps, seed=es)["avg_delay"])
        val_perf[a] = float(np.mean(delays))

    best_alpha = min(val_perf, key=val_perf.get)
    print(f"  Alpha selection on validation seeds {val_seeds}: {val_perf} -> Best alpha = {best_alpha}")

    # Evaluate best alpha on test seeds across scenarios
    item5_stats = {"val_performance": val_perf, "selected_alpha": best_alpha, "test_scenarios": {}}
    for sc in scenarios:
        plain_reps = [item1_data[sc]["M1_plain"][s] for s in training_seeds]
        bse_reps = [item1_data[sc]["M3_nonrec"][s] for s in training_seeds]
        
        # Test chosen alpha on test seeds
        ema_reps = []
        for s in training_seeds:
            c = EMAFilter(ctrl=load_clean_ippo(s), alpha=best_alpha)
            delays = []
            for es in test_seeds:
                env = SUMOEnv(net_file=cfg.net_file, route_file=cfg.route_file, num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=es)
                t = Trainer(env=env)
                c.reset(agent_ids)
                delays.append(t.run_evaluation_episode(c, max_steps=cfg.max_steps, seed=es)["avg_delay"])
            ema_reps.append(float(np.mean(delays)))

        diff_ema_plain = np.array(ema_reps) - np.array(plain_reps)
        diff_ema_bse = np.array(ema_reps) - np.array(bse_reps)
        ci_plain = compute_bootstrap_ci(diff_ema_plain)
        ci_bse = compute_bootstrap_ci(diff_ema_bse)

        # Significant if CI excludes zero
        sig_plain = bool(ci_plain[0] > 0 or ci_plain[1] < 0)
        sig_bse = bool(ci_bse[0] > 0 or ci_bse[1] < 0)

        item5_stats["test_scenarios"][sc] = {
            "plain_mean": float(np.mean(plain_reps)),
            "bse_mean": float(np.mean(bse_reps)),
            "ema_mean": float(np.mean(ema_reps)),
            "diff_ema_vs_plain": {"mean": float(np.mean(diff_ema_plain)), "ci_95": ci_plain, "significant": sig_plain},
            "diff_ema_vs_bse": {"mean": float(np.mean(diff_ema_bse)), "ci_95": ci_bse, "significant": sig_bse},
            "replicates_ema": ema_reps
        }
    results["item5"] = item5_stats

    # Save all output JSON
    res_path = os.path.join(out_dir, "diagnostic_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to -> {res_path}")

if __name__ == "__main__":
    main()
