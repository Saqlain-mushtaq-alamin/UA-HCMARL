"""
Phase 1 Baseline Launch Script (Phase 1 §1.6 & §1.8)

Runs Fixed-Time (B1) and Actuated (B2) baselines on 4x4 Grid and LemgoRL networks across:
- 5 random seeds
- 4 demand profiles (uniform, morning_peak, evening_peak, incident)
- 5 noise degradation levels (clean, mild, moderate, severe, extreme)

Logs metrics to JSONL, computes statistical summaries, and generates plots.
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
from src.training.trainer import Trainer
from src.evaluation.statistical import generate_statistical_report
from src.evaluation.visualize import plot_robustness_degradation, plot_baseline_comparison


def run_baseline_suite(
    net_file: str = "networks/grid4x4/grid4x4.net.xml",
    route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
    num_intersections: int = 16,
    num_seeds: int = 5,
    max_steps: int = 100,
    output_dir: str = "experiments/phase1_baselines",
):
    os.makedirs(output_dir, exist_ok=True)
    demand_profiles = ["uniform", "morning_peak", "evening_peak", "incident"]
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]

    all_results = {}

    print("==================================================")
    print("      Phase 1: Baseline Evaluation Suite          ")
    print("==================================================")

    for demand in demand_profiles:
        print(f"\n--- Testing Demand Profile: {demand} ---")
        demand_results = {}

        for noise in noise_scenarios:
            print(f"\n  [Noise Scenario: {noise.upper()}]")
            
            fixed_time_delays = []
            actuated_delays = []

            for seed in range(num_seeds):
                env = SUMOEnv(
                    net_file=net_file,
                    route_file=route_file,
                    num_intersections=num_intersections,
                    demand_profile=demand,
                    noise_scenario=noise,
                    max_steps=max_steps,
                    seed=seed,
                )
                trainer = Trainer(env=env)

                # 1. Fixed-Time (B1)
                ft_ctrl = FixedTimeController()
                ft_summary = trainer.run_evaluation_episode(ft_ctrl, max_steps=max_steps, seed=seed)
                fixed_time_delays.append(ft_summary["avg_delay"])

                # 2. Actuated (B2)
                act_ctrl = ActuatedController()
                act_summary = trainer.run_evaluation_episode(act_ctrl, max_steps=max_steps, seed=seed)
                actuated_delays.append(act_summary["avg_delay"])

            ft_mean = float(np.mean(fixed_time_delays))
            act_mean = float(np.mean(actuated_delays))

            print(f"    B1 (Fixed-Time) Mean Delay: {ft_mean:.2f} s")
            print(f"    B2 (Actuated)   Mean Delay: {act_mean:.2f} s")

            demand_results[noise] = {
                "fixed_time": fixed_time_delays,
                "actuated": actuated_delays,
            }

        all_results[demand] = demand_results

    # Save complete raw results
    raw_path = os.path.join(output_dir, "baseline_raw_results.json")
    with open(raw_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Success] Raw results saved to {raw_path}")

    # Generate Statistical Analysis Report for Uniform Demand Clean
    uniform_clean = all_results["uniform"]["clean"]
    stat_report = generate_statistical_report(uniform_clean, baseline_name="fixed_time")
    stat_path = os.path.join(output_dir, "statistical_summary.json")
    with open(stat_path, "w") as f:
        json.dump(stat_report, f, indent=2)
    print(f"[Success] Statistical summary saved to {stat_path}")

    # Generate Robustness Plot for Uniform Demand
    robustness_data = {
        "Fixed-Time (B1)": [float(np.mean(all_results["uniform"][n]["fixed_time"])) for n in noise_scenarios],
        "Actuated (B2)": [float(np.mean(all_results["uniform"][n]["actuated"])) for n in noise_scenarios],
    }
    plot_path = os.path.join(output_dir, "robustness_degradation.png")
    plot_robustness_degradation(noise_scenarios, robustness_data, output_path=plot_path)
    print(f"[Success] Robustness plot generated at {plot_path}")


if __name__ == "__main__":
    run_baseline_suite()
