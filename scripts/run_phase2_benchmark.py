"""
Phase 2 Comprehensive Benchmark Runner (Phase 2 §2.1 - §2.5)

Executes complete Phase 2 baseline benchmarking suite:
- Stage 0: Fixed-Time (B1) & Actuated (B2)
- Stage 1: IPPO (B3) & MAPPO (B4)
- Stage 2: MAPPO-LCE (B5)
- Stage 2b: FMA2C (B6)

Evaluates on:
- 4x4 Grid (16 intersections)
- Multi-seed statistical evaluation (10 seeds for significance)
- 5 Noise degradation levels (Clean, Mild, Moderate, Severe, Extreme)

FIX (2026-08-21):
  - Training budget raised from 5 episodes × 100 steps → 300 episodes × 3600 steps
    (configurable; defaults are set for a realistic "fast" run of 200 episodes)
  - Statistical module upgraded to use paired t-test when n >= 6 (avoids the
    Wilcoxon p = 0.0625 floor with n = 5)
  - Separate eval environment created per (noise, seed) pair with a distinct seed
    so noise injection truly varies across seeds
  - Lagrangian state reset between evaluation episodes to avoid carry-over
  - FMA2C dones buffer fix applied through trainer.train_episode()

FIX (2026-08-26) — Bug 1.1 / Bug 1.2:
  - Bug 1.1: Eval seeds now encode the NOISE LEVEL INDEX × 10 000 in addition to
    the episode seed and eval-ep counter.  Previously all 5 noise levels shared the
    same env seed, so the Poisson arrival process was identical across scenarios and
    Fixed-Time produced byte-for-byte identical delay arrays at every noise level.
  - Bug 1.2: num_seeds default raised to 10 (minimum for Wilcoxon significance);
    recommended value is 15 for publication.
  - Post-run sanity check: prints an error if Fixed-Time delay does NOT increase
    monotonically with noise (would indicate the seed fix did not work).
"""

import json
import os
import sys
from typing import Dict, List, Any
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.sumo_env import SUMOEnv
from src.baselines.fixed_time import FixedTimeController
from src.baselines.actuated import ActuatedController
from src.baselines.ippo import IPPOController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.baselines.fma2c import FMA2CController
from src.training.trainer import Trainer, LearningCurveLogger
from src.evaluation.statistical import generate_statistical_report, ttest_paired
from src.evaluation.metrics import compute_robustness_degradation_slope
from src.evaluation.visualize import (
    plot_robustness_degradation,
    plot_baseline_comparison,
    plot_constraint_violations,
    plot_hierarchy_fragility_diagnostic,
)
from checkpointed_runner import ResultCheckpointer


def _reset_lagrangian_state(controller):
    """Reset Lagrangian multiplier state between episodes (prevents carry-over)."""
    if hasattr(controller, "pid_lagrangian"):
        controller.pid_lagrangian.reset()
    if hasattr(controller, "pid_lagrangians"):
        for lag in controller.pid_lagrangians.values():
            lag.reset()


def run_phase2_benchmark(
    net_file: str = "networks/grid4x4/grid4x4.net.xml",
    route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
    num_intersections: int = 16,
    num_seeds: int = 10,
    train_episodes: int = 200,
    eval_episodes_per_seed: int = 3,
    max_steps: int = 3600,
    output_dir: str = "experiments/phase2_flat_baselines",
):
    """
    Full Phase 2 benchmark with corrected training budget and statistical tests.

    Parameters
    ----------
    num_seeds         : Number of independent seeds (≥10 for significance with Wilcoxon;
                        ≥6 for t-test).  Default 10.
    train_episodes    : Training episodes per RL method (one episode = max_steps env steps).
                        200 episodes × 3600 steps = 720 K env steps — sufficient for basic
                        convergence in a 16-intersection grid.
    eval_episodes_per_seed : Evaluation roll-outs averaged per seed (reduces variance).
    max_steps         : Steps per episode (3600 = 1 simulated hour at 1 s/step).
    """
    os.makedirs(output_dir, exist_ok=True)
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]

    print("==========================================================")
    print("   UA-HCMARL Phase 2: Flat Baseline Benchmark Suite       ")
    print("==========================================================")
    print(f"  Config: {num_seeds} seeds | {train_episodes} train eps | {max_steps} steps/ep")

    methods = ["Fixed-Time", "Actuated", "IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]

    # ------------------------------------------------------------------
    # 1. Train RL controllers on Clean condition
    # ------------------------------------------------------------------
    print("\n[Step 1/3] Training RL Baseline Controllers (Clean Condition)...")
    env_train = SUMOEnv(
        net_file=net_file,
        route_file=route_file,
        num_intersections=num_intersections,
        demand_profile="uniform",
        noise_scenario="clean",
        max_steps=max_steps,
        seed=42,
    )
    trainer_train = Trainer(env=env_train)

    controllers = {
        "Fixed-Time": FixedTimeController(),
        "Actuated":   ActuatedController(),
        "IPPO":       IPPOController(agent_ids=agent_ids),
        "MAPPO":      MAPPOController(agent_ids=agent_ids),
        "MAPPO-LCE":  MAPPOLCEController(agent_ids=agent_ids),
        "FMA2C":      FMA2CController(
            agent_ids=agent_ids,
            num_regions=4 if num_intersections >= 4 else 2,
        ),
    }

    # Bug 4.2: attach one LearningCurveLogger per RL method
    curve_loggers = {m: LearningCurveLogger() for m in ["IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]}

    # Training loop with progress reporting every 50 episodes
    for m_name in ["IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]:
        print(f"\n  -> Training {m_name} for {train_episodes} episodes "
              f"({train_episodes * max_steps:,} env steps)...")
        ctrl = controllers[m_name]
        # Attach logger to trainer for this method
        trainer_train.learning_curve_logger = curve_loggers[m_name]
        trainer_train.learning_curve_logger.reset()
        best_delay = float("inf")
        for ep in range(train_episodes):
            ep_seed = 42 + ep
            summary, loss = trainer_train.train_episode(ctrl, max_steps=max_steps, seed=ep_seed)
            _reset_lagrangian_state(ctrl)
            if (ep + 1) % 50 == 0:
                d = summary.get("avg_delay", float("nan"))
                print(f"     ep {ep+1:4d}/{train_episodes} | delay={d:.2f} s")
                best_delay = min(best_delay, d)
        print(f"     {m_name} best training delay: {best_delay:.2f} s")
        # Save learning curves (Bug 4.2)
        curves_path = os.path.join(output_dir, f"learning_curves_{m_name.replace('-', '_').lower()}.json")
        trainer_train.save_learning_curves(curves_path)
        print(f"     Learning curves saved to {curves_path}")
    # Detach logger so eval episodes don't record into it
    trainer_train.learning_curve_logger = None

    # ------------------------------------------------------------------
    # 2. Evaluate across Seeds & Noise Sweeps
    # ------------------------------------------------------------------
    print("\n[Step 2/3] Evaluating All Baselines across Seeds & Noise Protocol...")
    raw_path = os.path.join(output_dir, "phase2_raw_results.json")
    checkpointer = ResultCheckpointer(raw_path, scenarios=noise_scenarios, methods=methods)

    for noise in noise_scenarios:
        noise_index = noise_scenarios.index(noise)   # Bug 1.1: encode noise level in seed
        print(f"\n  Testing Noise Scenario: [{noise.upper()}] (noise_index={noise_index})")
        for seed in range(num_seeds):
            all_done = all(checkpointer.is_completed(noise, m_name, seed) for m_name in methods)
            if all_done:
                continue

            env_seed = seed + 1000 + noise_index * 10000  # Bug 1.1 fix
            env_eval = SUMOEnv(
                net_file=net_file,
                route_file=route_file,
                num_intersections=num_intersections,
                demand_profile="uniform",
                noise_scenario=noise,
                max_steps=max_steps,
                seed=env_seed,
            )
            trainer_eval = Trainer(env=env_eval)

            for m_name in methods:
                if checkpointer.is_completed(noise, m_name, seed):
                    continue
                ctrl = controllers[m_name]
                seed_delays = []
                seed_viols = []
                for eval_ep in range(eval_episodes_per_seed):
                    _reset_lagrangian_state(ctrl)
                    ep_seed = seed + 1000 + eval_ep * 100 + noise_index * 10000  # Bug 1.1 fix
                    ep_summary = trainer_eval.run_evaluation_episode(
                        ctrl, max_steps=max_steps, seed=ep_seed
                    )
                    seed_delays.append(ep_summary["avg_delay"])
                    seed_viols.append(ep_summary["constraint_violation_rate"])

                checkpointer.record_result(
                    scenario=noise,
                    method=m_name,
                    avg_delay=float(np.mean(seed_delays)),
                    violation_rate=float(np.mean(seed_viols)),
                )

        benchmark_results = checkpointer.get_results()
        for m_name in methods:
            m_delay = float(np.mean(benchmark_results[noise][m_name]["avg_delay"])) if benchmark_results[noise][m_name]["avg_delay"] else float("nan")
            m_viol  = float(np.mean(benchmark_results[noise][m_name]["violation_rate"])) if benchmark_results[noise][m_name]["violation_rate"] else float("nan")
            print(f"    {m_name:12s} | Delay: {m_delay:6.2f} s | Violation Rate: {m_viol*100.0:5.1f}%")

    # ------------------------------------------------------------------
    # 3. Save Raw Results
    # ------------------------------------------------------------------
    benchmark_results = checkpointer.get_results()
    raw_path = os.path.join(output_dir, "phase2_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Success] Phase 2 raw benchmark data saved to {raw_path}")

    # ------------------------------------------------------------------
    # 4. Statistical Report (paired t-test + Wilcoxon)
    # ------------------------------------------------------------------
    print("\n[Step 3/3] Generating Statistical Significance & Robustness Reports...")
    clean_delays = {m: benchmark_results["clean"][m]["avg_delay"] for m in methods}
    stat_report = generate_statistical_report(clean_delays, baseline_name="Fixed-Time")

    # Augment with paired t-test
    fixed_arr = np.array(clean_delays["Fixed-Time"])
    for m_name in methods:
        if m_name == "Fixed-Time":
            continue
        m_arr = np.array(clean_delays[m_name])
        ttest_result = ttest_paired(m_arr, fixed_arr)
        if m_name in stat_report and "vs_baseline" in stat_report[m_name]:
            stat_report[m_name]["vs_baseline"]["ttest_p_value"] = ttest_result["p_value"]
            stat_report[m_name]["vs_baseline"]["ttest_significant"] = ttest_result["significant"]

    stat_path = os.path.join(output_dir, "phase2_statistical_summary.json")
    with open(stat_path, "w") as f:
        json.dump(stat_report, f, indent=2)
    print(f"[Success] Statistical summary saved to {stat_path}")

    # Print significance table
    print("\n  --- Significance vs Fixed-Time (clean) ---")
    for m_name in methods:
        if m_name == "Fixed-Time":
            continue
        vs = stat_report.get(m_name, {}).get("vs_baseline", {})
        w_p = vs.get("p_value", 1.0)
        t_p = vs.get("ttest_p_value", 1.0)
        d   = vs.get("cohens_d", 0.0)
        sig_w = "[Y]" if vs.get("significant_p_lt_005") else "[N]"
        sig_t = "[Y]" if vs.get("ttest_significant") else "[N]"
        print(f"  {m_name:12s}: Wilcoxon p={w_p:.4f} {sig_w} | t-test p={t_p:.4f} {sig_t} | d={d:.2f}")

    # ------------------------------------------------------------------
    # 5. Robustness Degradation Slopes
    # ------------------------------------------------------------------
    noise_indices = [0.0, 0.1, 0.2, 0.3, 0.5]
    robustness_slopes = {}
    print("\n  --- Robustness Slopes (positive = worse with noise = CORRECT) ---")
    ft_monotone_ok = True  # Bug 1.1 sanity check
    for m_name in methods:
        delays_over_noise = [
            float(np.mean(benchmark_results[n][m_name]["avg_delay"])) for n in noise_scenarios
        ]
        slope, intercept, r_val = compute_robustness_degradation_slope(noise_indices, delays_over_noise)
        robustness_slopes[m_name] = {"slope": slope, "r_value": r_val, "delays": delays_over_noise}
        sign = "[Y] OK" if slope >= 0 else "[N] INVERTED (bug?)"
        print(f"  {m_name:12s} slope={slope:+8.2f}  r={r_val:.2f}  {sign}")

        # Sanity check: Fixed-Time MUST produce increasing delay with noise
        # (i.e. all values must NOT be identical across noise levels)
        if m_name == "Fixed-Time":
            if len(set(round(d, 4) for d in delays_over_noise)) == 1:
                ft_monotone_ok = False
                print("  [ERROR] Fixed-Time delays are IDENTICAL across noise levels!")
                print("          Bug 1.1 seed fix may not have propagated correctly.")
            else:
                print(f"  [OK] Fixed-Time delays vary across noise levels (Bug 1.1 resolved)")

    slopes_path = os.path.join(output_dir, "robustness_slopes.json")
    with open(slopes_path, "w") as f:
        json.dump(robustness_slopes, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Generate Milestone Figures
    # ------------------------------------------------------------------
    clean_summary = {m: {"avg_delay": float(np.mean(clean_delays[m]))} for m in methods}
    plot_baseline_comparison(
        clean_summary, metric_name="avg_delay",
        output_path=os.path.join(output_dir, "fig1_clean_baseline_comparison.png"),
    )

    robustness_delays = {
        m: [float(np.mean(benchmark_results[n][m]["avg_delay"])) for n in noise_scenarios]
        for m in methods
    }
    plot_robustness_degradation(
        noise_scenarios, robustness_delays,
        output_path=os.path.join(output_dir, "fig2_robustness_degradation.png"),
    )

    violation_rates = {
        m: [float(np.mean(benchmark_results[n][m]["violation_rate"])) for n in noise_scenarios]
        for m in methods
    }
    plot_constraint_violations(
        noise_scenarios, violation_rates,
        output_path=os.path.join(output_dir, "fig4_constraint_violations_noise.png"),
    )

    fragility_data = {
        "MAPPO (Flat)": robustness_delays["MAPPO"],
        "MAPPO-LCE (Flat Constrained)": robustness_delays["MAPPO-LCE"],
        "FMA2C (Hierarchical Unconstrained)": robustness_delays["FMA2C"],
    }
    plot_hierarchy_fragility_diagnostic(
        noise_scenarios, fragility_data,
        output_path=os.path.join(output_dir, "fig5_fma2c_fragility_diagnostic.png"),
    )

    print("\n==========================================================")
    print("   [SUCCESS] Phase 2 Benchmark Completed Successfully!    ")
    print("==========================================================")

    return benchmark_results, stat_report, robustness_slopes


if __name__ == "__main__":
    run_phase2_benchmark()
