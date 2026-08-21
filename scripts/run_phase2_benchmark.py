"""
Phase 2 Comprehensive Benchmark Runner (Phase 2 §2.1 - §2.5)

Executes complete Phase 2 baseline benchmarking suite:
- Stage 0: Fixed-Time (B1) & Actuated (B2)
- Stage 1: IPPO (B3) & MAPPO (B4)
- Stage 2: MAPPO-LCE (B5)
- Stage 2b: FMA2C (B6)

Evaluates on:
- 4x4 Grid (16 intersections) & LemgoRL (5 intersections)
- Multi-seed statistical evaluation (5 seeds)
- 5 Noise degradation levels (Clean, Mild, Moderate, Severe, Extreme)

Generates:
- Raw evaluation outputs in JSON
- Statistical analysis reports (Wilcoxon test, 95% CIs, Cohen's d)
- 4 Publication-grade milestone figures
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
from src.training.trainer import Trainer
from src.evaluation.statistical import generate_statistical_report
from src.evaluation.metrics import compute_robustness_degradation_slope
from src.evaluation.visualize import (
    plot_robustness_degradation,
    plot_baseline_comparison,
    plot_constraint_violations,
    plot_hierarchy_fragility_diagnostic,
)


def run_phase2_benchmark(
    net_file: str = "networks/grid4x4/grid4x4.net.xml",
    route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
    num_intersections: int = 16,
    num_seeds: int = 5,
    train_episodes: int = 5,
    max_steps: int = 100,
    output_dir: str = "experiments/phase2_flat_baselines",
):
    os.makedirs(output_dir, exist_ok=True)
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]

    print("==========================================================")
    print("   UA-HCMARL Phase 2: Flat Baseline Benchmark Suite       ")
    print("==========================================================")

    methods = ["Fixed-Time", "Actuated", "IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]

    # 1. Train RL controllers on Clean condition first
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
        "Actuated": ActuatedController(),
        "IPPO": IPPOController(agent_ids=agent_ids),
        "MAPPO": MAPPOController(agent_ids=agent_ids),
        "MAPPO-LCE": MAPPOLCEController(agent_ids=agent_ids),
        "FMA2C": FMA2CController(agent_ids=agent_ids, num_regions=4 if num_intersections >= 4 else 2),
    }

    # Perform brief training iterations for RL baselines
    for m_name in ["IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]:
        print(f"  -> Training {m_name} for {train_episodes} episodes...")
        ctrl = controllers[m_name]
        for ep in range(train_episodes):
            trainer_train.train_episode(ctrl, max_steps=max_steps, seed=42 + ep)

    # 2. Evaluate across Seeds & Noise Sweeps
    print("\n[Step 2/3] Evaluating All Baselines across Seeds & Noise Protocol...")
    benchmark_results: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        noise: {m: {"avg_delay": [], "violation_rate": []} for m in methods}
        for noise in noise_scenarios
    }

    for noise in noise_scenarios:
        print(f"\n  Testing Noise Scenario: [{noise.upper()}]")
        for seed in range(num_seeds):
            env_eval = SUMOEnv(
                net_file=net_file,
                route_file=route_file,
                num_intersections=num_intersections,
                demand_profile="uniform",
                noise_scenario=noise,
                max_steps=max_steps,
                seed=seed,
            )
            trainer_eval = Trainer(env=env_eval)

            for m_name in methods:
                ctrl = controllers[m_name]
                summary = trainer_eval.run_evaluation_episode(ctrl, max_steps=max_steps, seed=seed)
                benchmark_results[noise][m_name]["avg_delay"].append(summary["avg_delay"])
                benchmark_results[noise][m_name]["violation_rate"].append(summary["constraint_violation_rate"])

        for m_name in methods:
            m_delay = float(np.mean(benchmark_results[noise][m_name]["avg_delay"]))
            m_viol = float(np.mean(benchmark_results[noise][m_name]["violation_rate"]))
            print(f"    {m_name:12s} | Delay: {m_delay:6.2f} s | Violation Rate: {m_viol*100.0:5.1f}%")

    # 3. Save Raw Results
    raw_path = os.path.join(output_dir, "phase2_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Success] Phase 2 raw benchmark data saved to {raw_path}")

    # 4. Statistical Report on Clean Condition
    print("\n[Step 3/3] Generating Statistical Significance & Robustness Reports...")
    clean_delays = {m: benchmark_results["clean"][m]["avg_delay"] for m in methods}
    stat_report = generate_statistical_report(clean_delays, baseline_name="Fixed-Time")
    stat_path = os.path.join(output_dir, "phase2_statistical_summary.json")
    with open(stat_path, "w") as f:
        json.dump(stat_report, f, indent=2)
    print(f"[Success] Statistical summary saved to {stat_path}")

    # 5. Compute Robustness Degradation Slopes
    noise_indices = [0.0, 0.1, 0.2, 0.3, 0.5]
    robustness_slopes = {}
    for m_name in methods:
        delays_over_noise = [
            float(np.mean(benchmark_results[n][m_name]["avg_delay"])) for n in noise_scenarios
        ]
        slope, intercept, r_val = compute_robustness_degradation_slope(noise_indices, delays_over_noise)
        robustness_slopes[m_name] = {"slope": slope, "r_value": r_val}
        print(f"  * {m_name:12s} Robustness Slope (d_delay/d_noise): {slope:6.2f} (r={r_val:.2f})")

    slopes_path = os.path.join(output_dir, "robustness_slopes.json")
    with open(slopes_path, "w") as f:
        json.dump(robustness_slopes, f, indent=2)

    # 6. Generate Milestone Figures
    # Fig 1: Clean Baseline Comparison
    clean_summary = {m: {"avg_delay": float(np.mean(clean_delays[m]))} for m in methods}
    plot_baseline_comparison(clean_summary, metric_name="avg_delay", output_path=os.path.join(output_dir, "fig1_clean_baseline_comparison.png"))

    # Fig 2: Robustness Degradation Curves (Delay vs Noise)
    robustness_delays = {
        m: [float(np.mean(benchmark_results[n][m]["avg_delay"])) for n in noise_scenarios]
        for m in methods
    }
    plot_robustness_degradation(noise_scenarios, robustness_delays, output_path=os.path.join(output_dir, "fig2_robustness_degradation.png"))

    # Fig 3 & 4: Constraint Violation Rates under Noise
    violation_rates = {
        m: [float(np.mean(benchmark_results[n][m]["violation_rate"])) for n in noise_scenarios]
        for m in methods
    }
    plot_constraint_violations(noise_scenarios, violation_rates, output_path=os.path.join(output_dir, "fig4_constraint_violations_noise.png"))

    # Fig 5: FMA2C Fragility Diagnostic (Hierarchy Amplifies Noise Sensitivity)
    fragility_data = {
        "MAPPO (Flat)": robustness_delays["MAPPO"],
        "MAPPO-LCE (Flat Constrained)": robustness_delays["MAPPO-LCE"],
        "FMA2C (Hierarchical Unconstrained)": robustness_delays["FMA2C"],
    }
    plot_hierarchy_fragility_diagnostic(noise_scenarios, fragility_data, output_path=os.path.join(output_dir, "fig5_fma2c_fragility_diagnostic.png"))

    print("\n==========================================================")
    print("   [SUCCESS] Phase 2 Benchmark Completed Successfully!    ")
    print("==========================================================")


if __name__ == "__main__":
    run_phase2_benchmark()
