"""
Noise Protocol V2 Re-Evaluation Suite
=====================================
Evaluates ALL Phase 2 controllers (Fixed-Time, Actuated, IPPO, MAPPO, MAPPO-LCE, FMA2C)
across all 3 training seeds (42, 43, 44) under Noise Protocol V2.
Uses the same 10 base eval seeds x 3 episodes (30 episodes per model/scenario)
matching the Phase 2 canonical benchmark.
Also reconciles Recursive-BSE severe results across all 10 base seeds.
"""

import os
import sys
import json
import time
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.envs.sumo_env import SUMOEnv
from src.envs.noise_injection_v2 import NoiseInjectorV2
from src.baselines.fixed_time import FixedTimeController
from src.baselines.actuated import ActuatedController
from src.baselines.ippo import IPPOController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_wrapper import BSEControllerWrapper
from src.models.belief_encoder import GRUKalmanNet
from src.training.trainer import Trainer
from experiment_config import get_config

# ---------------------------------------------------------------------------
# SumoEnvV2 subclass
# ---------------------------------------------------------------------------
class SumoEnvV2(SUMOEnv):
    """SUMOEnv subclass using NoiseInjectorV2 (detector-only degradation)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_injector = NoiseInjectorV2(
            scenario=self.noise_scenario,
            num_intersections=self.num_intersections,
            seed=self._init_seed
        )

# ---------------------------------------------------------------------------
# Helper: instantiate controller
# ---------------------------------------------------------------------------
def create_eval_controller(m_name: str, agent_ids: List[str], num_intersections: int, model_path: Optional[str] = None):
    if m_name == "Fixed-Time":
        return FixedTimeController()
    elif m_name == "Actuated":
        return ActuatedController(queue_threshold=1.5, max_queue=20.0)
    elif m_name == "IPPO":
        ctrl = IPPOController(agent_ids=agent_ids, share_actor=True, share_critic=True)
    elif m_name == "MAPPO":
        ctrl = MAPPOController(agent_ids=agent_ids)
    elif m_name == "MAPPO-LCE":
        ctrl = MAPPOLCEController(agent_ids=agent_ids)
    elif m_name == "FMA2C":
        ctrl = FMA2CController(agent_ids=agent_ids, num_regions=4 if num_intersections >= 4 else 2)
    elif m_name == "BSE_IPPO_Recursive":
        base_ctrl = IPPOController(agent_ids=agent_ids, share_actor=True, share_critic=True)
        if model_path and os.path.exists(model_path):
            base_ctrl.load_model(model_path)
        bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
        bse_ckpt = "experiments/phase3_canonical/models/bse.pt"
        data = torch.load(bse_ckpt, map_location="cpu", weights_only=False)
        bse.load_state_dict(data["bse_state"] if "bse_state" in data else data)
        return BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=5)
    else:
        raise ValueError(f"Unknown method {m_name}")

    if model_path and os.path.exists(model_path) and hasattr(ctrl, "load_model"):
        ctrl.load_model(model_path)
    return ctrl

# ---------------------------------------------------------------------------
# Worker task function
# ---------------------------------------------------------------------------
def eval_task_worker(task: Tuple):
    """
    Evaluates one (method, train_seed, scenario, base_seed_idx) across 3 episodes.
    Returns: (scenario, method, train_seed, base_seed_idx, mean_queue, list_of_ep_queues)
    """
    (
        noise_ver, scenario, method, train_seed, base_seed_idx,
        net_file, route_file, num_intersections, max_steps, model_path
    ) = task

    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    ep_queues = []

    # 3 episodes per base seed: base_seed_idx + 1000 + k * 100
    for k in range(3):
        ep_seed = base_seed_idx + 1000 + k * 100
        if noise_ver == "v2":
            env = SumoEnvV2(
                net_file=net_file, route_file=route_file,
                num_intersections=num_intersections, noise_scenario=scenario,
                max_steps=max_steps, seed=ep_seed
            )
        else:
            env = SUMOEnv(
                net_file=net_file, route_file=route_file,
                num_intersections=num_intersections, noise_scenario=scenario,
                max_steps=max_steps, seed=ep_seed
            )

        ctrl = create_eval_controller(method, agent_ids, num_intersections, model_path)
        trainer = Trainer(env=env)
        if hasattr(ctrl, "reset"):
            ctrl.reset(agent_ids)

        res = trainer.run_evaluation_episode(ctrl, max_steps=max_steps, seed=ep_seed)
        q = float(res.get("avg_queue_length", res.get("avg_delay", 0.0)))
        ep_queues.append(q)

    mean_q = float(np.mean(ep_queues))
    return scenario, method, train_seed, base_seed_idx, mean_q, ep_queues

# ---------------------------------------------------------------------------
# Bootstrap CI helper
# ---------------------------------------------------------------------------
def bootstrap_ci_over_traffic_seeds(traffic_seed_means: np.ndarray, num_boot: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Resamples over the 10 traffic arrival base seeds."""
    n = len(traffic_seed_means)
    if n <= 1:
        return float(np.mean(traffic_seed_means)), float(np.mean(traffic_seed_means))
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(num_boot):
        sample = rng.choice(traffic_seed_means, size=n, replace=True)
        boot.append(float(np.mean(sample)))
    return float(np.percentile(boot, 100 * (alpha / 2.0))), float(np.percentile(boot, 100 * (1.0 - alpha / 2.0)))

def main():
    print("=" * 80, flush=True)
    print("STARTING NOISE PROTOCOL V2 RE-EVALUATION SUITE", flush=True)
    print("=" * 80, flush=True)

    cfg = get_config("canonical")
    out_dir = "experiments/phase2_v2noise_eval"
    os.makedirs(out_dir, exist_ok=True)
    models_dir = "experiments/phase2_canonical_v2/models"

    scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    classical_methods = ["Fixed-Time", "Actuated"]
    rl_methods = ["IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]
    train_seeds = [42, 43, 44]

    # Build tasks for Phase 2 methods under V2 noise
    tasks = []
    for sc in scenarios:
        # Classical methods (no training seed, train_seed=None)
        for m in classical_methods:
            for b_idx in range(10):
                tasks.append((
                    "v2", sc, m, None, b_idx,
                    cfg.net_file, cfg.route_file, cfg.num_intersections,
                    cfg.max_steps, None
                ))

        # RL methods (seeds 42, 43, 44)
        for m in rl_methods:
            for ts in train_seeds:
                suffix = f"_s{ts}" if ts != 42 else ""
                model_name = f"{m.replace('-', '_').lower()}{suffix}.pt"
                m_path = os.path.join(models_dir, model_name)
                for b_idx in range(10):
                    tasks.append((
                        "v2", sc, m, ts, b_idx,
                        cfg.net_file, cfg.route_file, cfg.num_intersections,
                        cfg.max_steps, m_path
                    ))

    # Also add tasks for Item 4: Recursive BSE on Severe noise across all 10 base seeds
    for ts in train_seeds:
        suffix = f"_s{ts}" if ts != 42 else ""
        m_path = os.path.join(models_dir, f"ippo{suffix}.pt")
        for b_idx in range(10):
            tasks.append((
                "v1", "severe", "BSE_IPPO_Recursive", ts, b_idx,
                cfg.net_file, cfg.route_file, cfg.num_intersections,
                cfg.max_steps, m_path
            ))

    print(f"Total evaluation tasks generated: {len(tasks)} (running in parallel with 16 workers)...", flush=True)
    t0 = time.time()

    raw_results = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(eval_task_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            sc, m, ts, b_idx, mean_q, ep_qs = fut.result()
            key = f"{m}_s{ts}" if ts is not None else m
            raw_results.setdefault(sc, {}).setdefault(key, {})[b_idx] = {
                "mean_queue": mean_q,
                "episodes": ep_qs
            }
            completed += 1
            if completed % 50 == 0 or completed == len(tasks):
                elapsed = time.time() - t0
                rate = completed / max(1.0, elapsed)
                rem_sec = (len(tasks) - completed) / max(0.01, rate)
                print(f"  Progress: {completed}/{len(tasks)} ({completed*100/len(tasks):.1f}%) | Elapsed: {elapsed:.1f}s | ETA: {rem_sec:.1f}s | Last: [{sc.upper()}] {key} base_seed={b_idx} -> {mean_q:.2f} veh", flush=True)

    # Save raw evaluation results
    raw_path = os.path.join(out_dir, "v2_eval_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"\nRaw results successfully saved to -> {raw_path}", flush=True)

    # -----------------------------------------------------------------------
    # Load canonical V1 results for side-by-side comparison
    # -----------------------------------------------------------------------
    v1_raw_path = "experiments/phase2_canonical_v2/phase2_raw_results.json"
    with open(v1_raw_path) as f:
        v1_raw = json.load(f)

    # -----------------------------------------------------------------------
    # Aggregate and compute statistics
    # -----------------------------------------------------------------------
    summary = {
        "v2_results": {},
        "v1_vs_v2_comparison": {},
        "item4_recursive_bse_reconciliation": {},
        "item3_filter_headroom": {},
    }

    all_methods_canonical = ["Fixed-Time", "Actuated", "IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]

    for sc in scenarios:
        summary["v2_results"][sc] = {}
        for m in all_methods_canonical:
            if m in classical_methods:
                # 10 traffic seed values
                traffic_means = [raw_results[sc][m][b]["mean_queue"] for b in range(10)]
                ci_low, ci_high = bootstrap_ci_over_traffic_seeds(np.array(traffic_means))
                summary["v2_results"][sc][m] = {
                    "k_training_replicates": 1,
                    "mean": float(np.mean(traffic_means)),
                    "std": float(np.std(traffic_means, ddof=1)),
                    "ci_95": [ci_low, ci_high],
                    "traffic_seed_values": traffic_means,
                }
            else:
                # RL method with 3 training seeds (42, 43, 44)
                rep_means = []
                traffic_seed_ensemble = []
                per_seed_traffic_values = {}
                for ts in train_seeds:
                    key = f"{m}_s{ts}"
                    ts_means = [raw_results[sc][key][b]["mean_queue"] for b in range(10)]
                    rep_means.append(float(np.mean(ts_means)))
                    per_seed_traffic_values[str(ts)] = ts_means

                # For each traffic seed b in 0..9, average across replicates
                for b in range(10):
                    avg_over_reps_for_b = np.mean([raw_results[sc][f"{m}_s{ts}"][b]["mean_queue"] for ts in train_seeds])
                    traffic_seed_ensemble.append(float(avg_over_reps_for_b))

                ci_low, ci_high = bootstrap_ci_over_traffic_seeds(np.array(traffic_seed_ensemble))
                summary["v2_results"][sc][m] = {
                    "k_training_replicates": 3,
                    "replicate_means": rep_means,
                    "replicate_mean": float(np.mean(rep_means)),
                    "replicate_std": float(np.std(rep_means, ddof=1)),
                    "traffic_seed_ensemble_mean": float(np.mean(traffic_seed_ensemble)),
                    "traffic_seed_ensemble_std": float(np.std(traffic_seed_ensemble, ddof=1)),
                    "ci_95_traffic_resampled": [ci_low, ci_high],
                    "per_seed_traffic_values": per_seed_traffic_values,
                }

    # Clean to Extreme degradation comparison (Item 2)
    for m in all_methods_canonical:
        # V1 numbers from canonical
        v1_clean = float(np.mean(v1_raw["clean"][m]["avg_delay"]))
        v1_ext = float(np.mean(v1_raw["extreme"][m]["avg_delay"]))
        v1_deg = v1_ext - v1_clean

        # V2 numbers
        if m in classical_methods:
            v2_clean = summary["v2_results"]["clean"][m]["mean"]
            v2_ext = summary["v2_results"]["extreme"][m]["mean"]
        else:
            v2_clean = summary["v2_results"]["clean"][m]["traffic_seed_ensemble_mean"]
            v2_ext = summary["v2_results"]["extreme"][m]["traffic_seed_ensemble_mean"]
        v2_deg = v2_ext - v2_clean

        summary["v1_vs_v2_comparison"][m] = {
            "v1_clean": v1_clean, "v1_extreme": v1_ext, "v1_degradation": v1_deg,
            "v2_clean": v2_clean, "v2_extreme": v2_ext, "v2_degradation": v2_deg,
            "degradation_reduction": v1_deg - v2_deg,
        }

        # Filter headroom (Item 3)
        summary["item3_filter_headroom"][m] = {
            "clean_queue": v2_clean,
            "v2_extreme_queue": v2_ext,
            "filter_headroom": float(v2_deg)
        }

    # Item 4: Reconcile Recursive BSE severe
    item4_bse = {}
    for ts in train_seeds:
        key = f"BSE_IPPO_Recursive_s{ts}"
        bse_ts_means = [raw_results["severe"][key][b]["mean_queue"] for b in range(10)]
        item4_bse[str(ts)] = {
            "traffic_seed_means": bse_ts_means,
            "seed_1000_mean": bse_ts_means[0],
            "overall_replicate_mean": float(np.mean(bse_ts_means)),
            "per_episode_raw": [raw_results["severe"][key][b]["episodes"] for b in range(10)]
        }

    all_seed_1000_vals = [item4_bse[str(ts)]["seed_1000_mean"] for ts in train_seeds]
    all_rep_means = [item4_bse[str(ts)]["overall_replicate_mean"] for ts in train_seeds]
    summary["item4_recursive_bse_reconciliation"] = {
        "per_replicate": item4_bse,
        "seed_1000_average_across_reps": float(np.mean(all_seed_1000_vals)),
        "ten_seed_benchmark_mean": float(np.mean(all_rep_means)),
    }

    # Save summary json
    summary_path = os.path.join(out_dir, "v2_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to -> {summary_path}", flush=True)

if __name__ == "__main__":
    main()
