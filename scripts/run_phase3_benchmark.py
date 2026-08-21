"""
Phase 3 Benchmark & Validation Suite (Phase 3 §3.1 - §3.6)

Executes complete Layer 1 + Layer 2 Phase 3 pipeline:
- Phase A: BSE Supervised Pre-Training & Calibration Validation
- Phase B & C: Staged RL Curriculum Training (Frozen & Unfrozen BSE)
- Exp 3.1: Clean Performance Preservation
- Exp 3.2: Noise Robustness & Constraint Violation Sweep (Key Result)
- Exp 3.3: Confidence-Modulation Ablation (Novelty N3)
- Exp 3.4: BSE Estimation Quality & Uncertainty Calibration Report
"""

import json
import os
import sys
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo_lce import MAPPOLCEController
from src.baselines.bse_mappo_lce import BSEMAPPOLCEController
from src.training.trainer import Trainer
from src.training.bse_trainer import (
    collect_bse_dataset,
    train_bse_supervised,
    evaluate_bse_calibration,
)
from src.evaluation.statistical import generate_statistical_report
from src.evaluation.visualize import setup_plot_style, plot_robustness_degradation, plot_constraint_violations


def run_phase3_benchmark(
    net_file: str = "networks/grid4x4/grid4x4.net.xml",
    route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
    num_intersections: int = 16,
    num_seeds: int = 10,
    train_episodes: int = 100,
    max_steps: int = 3600,
    output_dir: str = "experiments/phase3_belief_encoder",
):
    os.makedirs(output_dir, exist_ok=True)
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]

    print("==========================================================")
    print("   UA-HCMARL Phase 3: Layer 1 Belief State Encoder Suite  ")
    print("==========================================================")

    # -----------------------------------------------------------------
    # Step 1: Pre-training Layer 1 GRU-KalmanNet (Phase A)
    # -----------------------------------------------------------------
    print("\n[Step 1/4] Training Phase A: GRU-KalmanNet Pre-Training...")
    env_data = SUMOEnv(
        net_file=net_file,
        route_file=route_file,
        num_intersections=num_intersections,
        max_steps=max_steps,
        seed=42,
    )
    print("  -> Collecting (o_noisy, s_true) training pairs across noise levels...")
    X_train, Y_train = collect_bse_dataset(env_data, num_steps=2000, noise_scenarios=noise_scenarios)
    X_test, Y_test = collect_bse_dataset(env_data, num_steps=500, noise_scenarios=noise_scenarios)

    # Instantiate Stage 3 controller & pre-train its BSE
    bse_mappo_lce = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=True, freeze_bse=False)
    bse_mappo_ablation = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=False, freeze_bse=False)
    vanilla_mappo_lce = MAPPOLCEController(agent_ids=agent_ids)

    print("  -> Optimizing BSE via L_BSE loss...")
    loss_history = train_bse_supervised(bse_mappo_lce.bse, X_train, Y_train, epochs=20, batch_size=64)
    train_bse_supervised(bse_mappo_ablation.bse, X_train, Y_train, epochs=20, batch_size=64)

    calib_metrics = evaluate_bse_calibration(bse_mappo_lce.bse, X_test, Y_test)
    print(f"  * Raw MSE: {calib_metrics['raw_mse']:.4f} | BSE MSE: {calib_metrics['bse_mse']:.4f} ({calib_metrics['mse_reduction_pct']:.1f}% reduction)")
    print(f"  * Sigma Calibration Correlation r(sigma, err): {calib_metrics['sigma_calibration_correlation']:.3f}")

    # -----------------------------------------------------------------
    # Step 2: Training Curriculum (Phases B & C)
    # -----------------------------------------------------------------
    print("\n[Step 2/4] Training Phases B & C: Staged Curriculum Fine-Tuning...")
    env_train = SUMOEnv(
        net_file=net_file,
        route_file=route_file,
        num_intersections=num_intersections,
        max_steps=max_steps,
        seed=42,
    )
    trainer = Trainer(env=env_train)

    print("  -> Fine-tuning BSE+MAPPO-LCE (Proposed)...")
    for ep in range(train_episodes):
        trainer.train_episode(bse_mappo_lce, max_steps=max_steps, seed=42 + ep)

    print("  -> Fine-tuning BSE+MAPPO-LCE (Ablation: No Sigma-Modulation)...")
    for ep in range(train_episodes):
        trainer.train_episode(bse_mappo_ablation, max_steps=max_steps, seed=42 + ep)

    print("  -> Fine-tuning Vanilla MAPPO-LCE (Baseline)...")
    for ep in range(train_episodes):
        trainer.train_episode(vanilla_mappo_lce, max_steps=max_steps, seed=42 + ep)

    # -----------------------------------------------------------------
    # Step 3: Comprehensive Multi-Seed Evaluation across Noise Protocol
    # -----------------------------------------------------------------
    print("\n[Step 3/4] Running Experiments 3.1, 3.2, 3.3 across Noise Protocol...")
    methods = {
        "MAPPO-LCE (Vanilla)": vanilla_mappo_lce,
        "BSE+MAPPO-LCE (No Modulation)": bse_mappo_ablation,
        "BSE+MAPPO-LCE (Proposed)": bse_mappo_lce,
    }

    results: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        noise: {m: {"avg_delay": [], "violation_rate": []} for m in methods.keys()}
        for noise in noise_scenarios
    }

    for noise in noise_scenarios:
        print(f"\n  Testing Noise Scenario: [{noise.upper()}]")
        for seed in range(num_seeds):
            env_eval = SUMOEnv(
                net_file=net_file,
                route_file=route_file,
                num_intersections=num_intersections,
                noise_scenario=noise,
                max_steps=max_steps,
                seed=seed + 1000,  # Offset from training seeds
            )
            trainer_eval = Trainer(env=env_eval)

            for m_name, ctrl in methods.items():
                summary = trainer_eval.run_evaluation_episode(
                    ctrl, max_steps=max_steps, seed=seed + 1000
                )
                results[noise][m_name]["avg_delay"].append(summary["avg_delay"])
                results[noise][m_name]["violation_rate"].append(summary["constraint_violation_rate"])

        for m_name in methods.keys():
            m_delay = float(np.mean(results[noise][m_name]["avg_delay"]))
            m_viol = float(np.mean(results[noise][m_name]["violation_rate"]))
            print(f"    {m_name:32s} | Delay: {m_delay:6.2f} s | Violation Rate: {m_viol*100.0:5.1f}%")

    # -----------------------------------------------------------------
    # Step 4: Statistical Analysis & Figure Artifact Generation
    # -----------------------------------------------------------------
    print("\n[Step 4/4] Generating Statistical Reports & Phase 3 Milestone Figures...")
    
    # Save raw benchmark data
    with open(os.path.join(output_dir, "phase3_raw_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Statistical summary on Clean vs Severe condition
    clean_delays = {m: results["clean"][m]["avg_delay"] for m in methods.keys()}
    stat_report = generate_statistical_report(clean_delays, baseline_name="MAPPO-LCE (Vanilla)")
    with open(os.path.join(output_dir, "phase3_statistical_summary.json"), "w") as f:
        json.dump(stat_report, f, indent=2)

    calib_path = os.path.join(output_dir, "bse_calibration_metrics.json")
    with open(calib_path, "w") as f:
        json.dump(calib_metrics, f, indent=2)

    # Fig 1: Clean Performance Preservation
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    methods_list = list(methods.keys())
    clean_means = [float(np.mean(results["clean"][m]["avg_delay"])) for m in methods_list]
    bars = ax.bar(methods_list, clean_means, color=["#4c72b0", "#dd8452", "#55a868"], width=0.5, edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.0, f"{bar.get_height():.1f}s", ha="center", va="bottom")
    ax.set_ylabel("Average Delay (s)")
    ax.set_title("Exp 3.1: Clean Condition Performance Preservation")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig1_clean_preservation.png"))
    plt.close()

    # Fig 2: Noise Robustness Curves (Delay vs Noise)
    delay_sweeps = {m: [float(np.mean(results[n][m]["avg_delay"])) for n in noise_scenarios] for m in methods_list}
    plot_robustness_degradation(noise_scenarios, delay_sweeps, output_path=os.path.join(output_dir, "fig2_robustness_noise_sweep.png"))

    # Fig 3: Constraint Violation Sweep (Novelty N3 Ablation)
    viol_sweeps = {m: [float(np.mean(results[n][m]["violation_rate"])) for n in noise_scenarios] for m in methods_list}
    plot_constraint_violations(noise_scenarios, viol_sweeps, output_path=os.path.join(output_dir, "fig3_sigma_modulation_ablation.png"))

    # Fig 4: BSE Calibration & Quality
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.bar(["Raw Obs MSE", "BSE Denoised MSE"], [calib_metrics["raw_mse"], calib_metrics["bse_mse"]], color=["#c44e52", "#55a868"], width=0.4, edgecolor="black")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title(f"Exp 3.4: BSE Denoising Quality ({calib_metrics['mse_reduction_pct']:.1f}% MSE Reduction)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig4_bse_calibration_quality.png"))
    plt.close()

    print("\n==========================================================")
    print("   [SUCCESS] Phase 3 Benchmark Completed Successfully!    ")
    print("==========================================================")


if __name__ == "__main__":
    run_phase3_benchmark()
