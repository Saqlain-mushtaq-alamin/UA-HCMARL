"""
Master Experiment Runner — UA-HCMARL Publication Results
=========================================================
Runs all 5 experimental phases sequentially and saves publication-ready
JSON data + figures to experiments/.

Usage:
    python scripts/run_full_experiment.py              # full run (~60 min)
    python scripts/run_full_experiment.py --fast       # fast run (~8 min)

Fast mode: 4 intersections, 80 train eps, 500 steps, 5 seeds
Full mode: 16 intersections, 200 train eps, 1000 steps, 10 seeds
"""

import sys, os, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.sumo_env import SUMOEnv
from src.baselines.fixed_time import FixedTimeController
from src.baselines.actuated import ActuatedController
from src.baselines.ippo import IPPOController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_mappo_lce import BSEMAPPOLCEController
from src.baselines.ua_hcmarl import UAHCMARLController
from src.training.trainer import Trainer
from src.training.bse_trainer import collect_bse_dataset, train_bse_supervised, evaluate_bse_calibration
from src.evaluation.statistical import generate_statistical_report
from src.evaluation.metrics import compute_robustness_degradation_slope

NOISE_SCENARIOS = ["clean", "mild", "moderate", "severe", "extreme"]
NOISE_LEVELS    = [0.0, 0.1, 0.2, 0.3, 0.5]


def _reset_lagrangian(ctrl):
    if hasattr(ctrl, "pid_lagrangian"):
        ctrl.pid_lagrangian.reset()
    if hasattr(ctrl, "pid_lagrangians"):
        for l in ctrl.pid_lagrangians.values():
            l.reset()


def make_env(num_intersections, noise="clean", seed=42, max_steps=3600):
    return SUMOEnv(
        num_intersections=num_intersections,
        noise_scenario=noise,
        max_steps=max_steps,
        seed=seed,
    )


def train_rl(ctrl, num_intersections, train_eps, max_steps, label=""):
    print(f"    Training {label} ({train_eps} eps x {max_steps} steps)...")
    env = make_env(num_intersections, seed=42, max_steps=max_steps)
    trainer = Trainer(env=env)
    delays = []
    for ep in range(train_eps):
        summary, _ = trainer.train_episode(ctrl, max_steps=max_steps, seed=42 + ep)
        _reset_lagrangian(ctrl)
        delays.append(summary["avg_delay"])
        if (ep + 1) % 20 == 0:
            print(f"      ep {ep+1:3d}/{train_eps}  delay={np.mean(delays[-10:]):.2f}")
    return delays


def eval_ctrl(ctrl, num_intersections, noise, seed, max_steps, n_eval=2):
    delays, viols = [], []
    for k in range(n_eval):
        ep_seed = seed * 100 + k + 5000
        env = make_env(num_intersections, noise=noise, seed=ep_seed, max_steps=max_steps)
        trainer = Trainer(env=env)
        _reset_lagrangian(ctrl)
        s = trainer.run_evaluation_episode(ctrl, max_steps=max_steps, seed=ep_seed)
        delays.append(s["avg_delay"])
        viols.append(s["constraint_violation_rate"])
    return float(np.mean(delays)), float(np.mean(viols))


def noise_sweep(controllers, num_intersections, num_seeds, max_steps, n_eval=2):
    results = {
        n: {m: {"avg_delay": [], "violation_rate": []} for m in controllers}
        for n in NOISE_SCENARIOS
    }
    for noise in NOISE_SCENARIOS:
        print(f"  Noise [{noise}]", end="", flush=True)
        for seed in range(num_seeds):
            print(".", end="", flush=True)
            for m_name, ctrl in controllers.items():
                d, v = eval_ctrl(ctrl, num_intersections, noise, seed, max_steps, n_eval)
                results[noise][m_name]["avg_delay"].append(d)
                results[noise][m_name]["violation_rate"].append(v)
        print()
    return results


def save(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved -> {path}")


def slopes(results, methods):
    out = {}
    for m in methods:
        vals = [np.mean(results[n][m]["avg_delay"]) for n in NOISE_SCENARIOS]
        sl, _, r = compute_robustness_degradation_slope(NOISE_LEVELS, vals)
        out[m] = {"slope": sl, "r_value": r, "delays": vals}
        sign = "OK" if sl >= 0 else "INVERTED"
        print(f"    {m:28s} slope={sl:+7.2f} r={r:.2f} [{sign}]")
    return out


# --- PHASE 2: Flat Baselines ---
# ─────────────────────────────────────────────
def run_phase2(cfg):
    print("\n" + "="*60)
    print("  PHASE 2: Flat Baseline Benchmark")
    print("="*60)
    n = cfg["num_intersections"]
    agent_ids = [f"intersection_{i}" for i in range(n)]
    n_regions = min(4, n)

    controllers = {
        "Fixed-Time": FixedTimeController(),
        "Actuated":   ActuatedController(),
        "IPPO":       IPPOController(agent_ids=agent_ids),
        "MAPPO":      MAPPOController(agent_ids=agent_ids),
        "MAPPO-LCE":  MAPPOLCEController(agent_ids=agent_ids),
        "FMA2C":      FMA2CController(agent_ids=agent_ids, num_regions=n_regions),
    }

    for name in ["IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]:
        train_rl(controllers[name], n, cfg["train_eps"], cfg["max_steps"], name)

    print("\n  Running noise sweep ...")
    results = noise_sweep(controllers, n, cfg["num_seeds"], cfg["max_steps"], cfg["n_eval"])

    out = "experiments/phase2_flat_baselines"
    save(results, f"{out}/phase2_raw_results.json")

    # Stats
    clean_delays = {m: results["clean"][m]["avg_delay"] for m in controllers}
    stat = generate_statistical_report(clean_delays, baseline_name="Fixed-Time")
    save(stat, f"{out}/phase2_statistical_summary.json")

    # Slopes
    print("  Robustness slopes (should be >=0 for every method):")
    sl = slopes(results, list(controllers.keys()))
    save(sl, f"{out}/robustness_slopes.json")

    # Print summary table
    print("\n  === Phase 2 Clean Results ===")
    print(f"  {'Method':14s} {'Delay':>10s} {'Viol%':>8s}")
    for m in controllers:
        d = np.mean(results["clean"][m]["avg_delay"])
        v = np.mean(results["clean"][m]["violation_rate"]) * 100
        print(f"  {m:14s} {d:10.2f} {v:8.1f}%")

    return results, controllers


# --- PHASE 3: BSE + MAPPO-LCE ---
# ─────────────────────────────────────────────
def run_phase3(cfg, phase2_controllers):
    print("\n" + "="*60)
    print("  PHASE 3: Belief-State Encoder (BSE) Integration")
    print("="*60)
    n = cfg["num_intersections"]
    agent_ids = [f"intersection_{i}" for i in range(n)]

    env_data = make_env(n, seed=42, max_steps=cfg["max_steps"])
    print("  Collecting BSE dataset ...")
    X_tr, Y_tr = collect_bse_dataset(env_data, num_steps=cfg["bse_steps"])
    X_te, Y_te = collect_bse_dataset(env_data, num_steps=cfg["bse_steps"] // 4)

    bse_proposed  = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=True)
    bse_ablation  = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=False)
    vanilla_lce   = phase2_controllers["MAPPO-LCE"]

    for ctrl, label in [(bse_proposed, "BSE+LCE (Proposed)"), (bse_ablation, "BSE+LCE (No-Mod)")]:
        print(f"  Pre-training BSE for {label} ...")
        train_bse_supervised(ctrl.bse, X_tr, Y_tr, epochs=cfg["bse_epochs"], batch_size=64)
        train_rl(ctrl, n, cfg["train_eps"], cfg["max_steps"], label)

    calib = evaluate_bse_calibration(bse_proposed.bse, X_te, Y_te)
    print(f"  BSE Calibration: raw_MSE={calib['raw_mse']:.4f} bse_MSE={calib['bse_mse']:.4f} "
          f"reduction={calib['mse_reduction_pct']:.1f}% corr={calib['sigma_calibration_correlation']:.3f}")

    controllers = {
        "MAPPO-LCE (Vanilla)":        vanilla_lce,
        "BSE+MAPPO-LCE (No-Mod)":    bse_ablation,
        "BSE+MAPPO-LCE (Proposed)":  bse_proposed,
    }
    results = noise_sweep(controllers, n, cfg["num_seeds"], cfg["max_steps"], cfg["n_eval"])

    out = "experiments/phase3_belief_encoder"
    save(results, f"{out}/phase3_raw_results.json")
    save(calib,   f"{out}/bse_calibration_metrics.json")

    stat = generate_statistical_report(
        {m: results["clean"][m]["avg_delay"] for m in controllers},
        baseline_name="MAPPO-LCE (Vanilla)"
    )
    save(stat, f"{out}/phase3_statistical_summary.json")

    print("\n  === Phase 3 Clean Results ===")
    print(f"  {'Method':30s} {'Delay':>10s}")
    for m in controllers:
        d = np.mean(results["clean"][m]["avg_delay"])
        print(f"  {m:30s} {d:10.2f}")

    return results, bse_proposed


# --- PHASE 4: Full UA-HCMARL ---
# ─────────────────────────────────────────────
def run_phase4(cfg, bse_proposed, phase2_controllers):
    print("\n" + "="*60)
    print("  PHASE 4: Full UA-HCMARL System")
    print("="*60)
    n = cfg["num_intersections"]
    agent_ids = [f"intersection_{i}" for i in range(n)]
    n_regions = min(4, n)

    ua_hcmarl = UAHCMARLController(agent_ids=agent_ids, num_regions=n_regions)
    train_rl(ua_hcmarl, n, cfg["train_eps"], cfg["max_steps"], "UA-HCMARL")

    controllers = {
        "Fixed-Time":     phase2_controllers["Fixed-Time"],
        "MAPPO":          phase2_controllers["MAPPO"],
        "BSE+MAPPO-LCE":  bse_proposed,
        "UA-HCMARL":      ua_hcmarl,
    }
    results = noise_sweep(controllers, n, cfg["num_seeds"], cfg["max_steps"], cfg["n_eval"])

    out = "experiments/phase4_full_hierarchy"
    save(results, f"{out}/phase4_raw_results.json")

    stat = generate_statistical_report(
        {m: results["clean"][m]["avg_delay"] for m in controllers},
        baseline_name="Fixed-Time"
    )
    save(stat, f"{out}/phase4_statistical_summary.json")

    print("\n  === Phase 4 Clean Results ===")
    print(f"  {'Method':28s} {'Delay':>10s} {'Viol%':>8s}")
    for m in controllers:
        d = np.mean(results["clean"][m]["avg_delay"])
        v = np.mean(results["clean"][m]["violation_rate"]) * 100
        print(f"  {m:28s} {d:10.2f} {v:8.1f}%")

    return results, ua_hcmarl


# --- PHASE 5: Controlled Ablation ---
# ─────────────────────────────────────────────
def run_phase5(cfg, ua_hcmarl, bse_proposed, phase2_controllers):
    print("\n" + "="*60)
    print("  PHASE 5: Controlled Ablation Suite")
    print("="*60)
    n = cfg["num_intersections"]
    agent_ids = [f"intersection_{i}" for i in range(n)]
    n_regions = min(4, n)

    # Ablation variants on clean condition only (fast)
    ablations = {
        "A0: UA-HCMARL (Full)":          ua_hcmarl,
        "A1: BSE+MAPPO-LCE":             bse_proposed,
        "A2: MAPPO-LCE (Vanilla)":       phase2_controllers["MAPPO-LCE"],
        "A3: MAPPO (No Constraints)":    phase2_controllers["MAPPO"],
        "A4: FMA2C (Hier, No BSE)":      phase2_controllers["FMA2C"],
        "A5: Fixed-Time":                phase2_controllers["Fixed-Time"],
    }

    clean_results = {}
    print("  Running clean-condition ablation ...")
    for m_name, ctrl in ablations.items():
        delays, viols = [], []
        for seed in range(cfg["num_seeds"]):
            d, v = eval_ctrl(ctrl, n, "clean", seed, cfg["max_steps"], cfg["n_eval"])
            delays.append(d); viols.append(v)
        clean_results[m_name] = {
            "avg_delay": delays,
            "violation_rate": viols,
            "mean_delay": float(np.mean(delays)),
            "mean_viol":  float(np.mean(viols)),
        }
        print(f"    {m_name:32s} delay={np.mean(delays):.2f}  viol={np.mean(viols)*100:.1f}%")

    out = "experiments/phase5_ablations"
    save(clean_results, f"{out}/phase5_ablation_results.json")

    stat = generate_statistical_report(
        {m: v["avg_delay"] for m, v in clean_results.items()},
        baseline_name="A5: Fixed-Time"
    )
    save(stat, f"{out}/phase5_statistical_summary.json")

    return clean_results


# --- ENTRY POINT ---
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        type=str,
        default="canonical",
        choices=["canonical", "fast", "smoke"],
        help="Configuration profile (default: canonical)",
    )
    parser.add_argument("--fast", action="store_true", help="Deprecated: shorthand for --profile fast")
    args = parser.parse_args()

    from experiment_config import get_config
    profile_to_use = "fast" if args.fast else args.profile
    cfg = get_config(profile_to_use).to_dict()
    print(f"Mode: {cfg['profile_name'].upper()} ({cfg['num_intersections']} intersections | {cfg['train_episodes']} train eps | {cfg['max_steps']} steps | {cfg['num_seeds']} seeds)")

    t0 = time.time()
    results_p2, ctrls_p2          = run_phase2(cfg)
    results_p3, bse_proposed      = run_phase3(cfg, ctrls_p2)
    results_p4, ua_hcmarl         = run_phase4(cfg, bse_proposed, ctrls_p2)
    results_p5                    = run_phase5(cfg, ua_hcmarl, bse_proposed, ctrls_p2)

    elapsed = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"  ALL PHASES COMPLETE in {elapsed:.1f} minutes")
    print(f"  Results saved to experiments/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
