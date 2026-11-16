"""
Phase 3 Benchmark & Validation Suite: Six-Arm Multi-Replicate Robustness Evaluation
===================================================================================
Evaluates Layer 1 Belief-State Estimation (BSE) against clean-trained and noise-trained
baselines across six controlled experimental arms and multiple independent training replicates:

  Arm M1: IPPO (Clean-Trained)
  Arm M2: IPPO (Noise-Trained) [Data Augmentation Baseline]
  Arm M3: IPPO + BSE (Ours)    [Latent State Estimation]
  Arm M4: FMA2C (Clean-Trained)
  Arm M5: FMA2C (Noise-Trained) [Hierarchical Data Augmentation]
  Arm M6: FMA2C + BSE (Ours)   [Hierarchical Latent Stabilization]

Evaluates queue length, degradation slope (beta), starvation constraint violations,
and throughput across 5 standardized noise scenarios (Clean, Mild, Moderate, Severe, Extreme)
on identical matched traffic arrival seeds.

Methodological Guarantees:
- Independent training replicates (seeds 42, 43, 44) form the unit of statistical analysis.
- Full Kalman recurrence: b_hat(t) = (1 - K(t)) * f(b_hat(t-1)) + K(t) * o(t).
- Complete controller instance isolation (zero cross-arm state pollution).
- Full cryptographic provenance (SHA-256 hashes of all artifacts and git revision tracking).
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import torch

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_wrapper import BSEControllerWrapper
from src.models.belief_encoder import GRUKalmanNet
from src.training.trainer import Trainer
from src.training.bse_trainer import (
    collect_bse_dataset,
    train_bse_supervised,
    evaluate_bse_calibration,
)
from src.training.noise_training import train_noise_augmented_controller
from src.evaluation.visualize import setup_plot_style, plot_robustness_degradation
from checkpointed_runner import ResultCheckpointer
from experiment_config import get_config, ExperimentConfig


def compute_sha256(filepath: str) -> str:
    """Computes SHA-256 hash of a file on disk."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_metadata() -> Dict[str, Any]:
    """Retrieves current git commit, branch, and working tree cleanliness."""
    metadata = {
        "commit": "unknown",
        "branch": "unknown",
        "dirty": False,
        "modified_files": [],
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        metadata["commit"] = commit
    except Exception:
        pass

    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        metadata["branch"] = branch
    except Exception:
        pass

    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode("utf-8").strip()
        if status:
            metadata["dirty"] = True
            metadata["modified_files"] = [line.strip() for line in status.splitlines()]
    except Exception:
        pass

    return metadata


def compute_replicate_statistics(
    replicate_means: List[float],
) -> Dict[str, Any]:
    """
    Computes rigorous statistical summary across independent training replicates.
    """
    K = len(replicate_means)
    if K == 0:
        return {"replicate_means": [], "mean": float("nan"), "std": float("nan"), "sem": float("nan"), "ci_95": [float("nan"), float("nan")]}

    arr = np.array(replicate_means, dtype=np.float64)
    mean_val = float(np.mean(arr))
    if K > 1:
        std_val = float(np.std(arr, ddof=1))
        sem_val = float(std_val / np.sqrt(K))
        # Student's t critical value for 95% two-sided CI with df = K-1
        t_crit = float(stats.t.ppf(0.975, df=K-1))
        ci_lower = mean_val - t_crit * sem_val
        ci_upper = mean_val + t_crit * sem_val
    else:
        std_val = 0.0
        sem_val = 0.0
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "k_replicates": K,
        "replicate_means": [float(x) for x in arr],
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "ci_95": [float(ci_lower), float(ci_upper)],
    }


def compute_paired_comparison(
    treatment_means: List[float],
    baseline_means: List[float],
) -> Dict[str, Any]:
    """
    Computes paired Student's t-test and Wilcoxon signed-rank test across training replicates.
    """
    assert len(treatment_means) == len(baseline_means), "Replicate count mismatch!"
    K = len(treatment_means)
    diffs = np.array(treatment_means, dtype=np.float64) - np.array(baseline_means, dtype=np.float64)
    mean_diff = float(np.mean(diffs))

    if K > 1:
        std_diff = float(np.std(diffs, ddof=1))
        sem_diff = float(std_diff / np.sqrt(K))
        t_stat, p_val_t = stats.ttest_rel(treatment_means, baseline_means)
        t_stat = float(t_stat)
        p_val_t = float(p_val_t)

        # Wilcoxon signed-rank test
        if np.all(diffs == 0.0):
            p_val_wilcoxon = 1.0
            w_stat = 0.0
        else:
            try:
                res_w = stats.wilcoxon(treatment_means, baseline_means, alternative="two-sided")
                w_stat = float(res_w.statistic)
                p_val_wilcoxon = float(res_w.pvalue)
            except Exception:
                w_stat = float("nan")
                p_val_wilcoxon = float("nan")

        t_crit = float(stats.t.ppf(0.975, df=K-1))
        ci_diff = [mean_diff - t_crit * sem_diff, mean_diff + t_crit * sem_diff]
    else:
        std_diff = 0.0
        sem_diff = 0.0
        t_stat = float("nan")
        p_val_t = float("nan")
        w_stat = float("nan")
        p_val_wilcoxon = float("nan")
        ci_diff = [mean_diff, mean_diff]

    return {
        "k_replicates": K,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "ci_95_difference": ci_diff,
        "t_statistic": t_stat,
        "p_value_t_test": p_val_t,
        "wilcoxon_statistic": w_stat,
        "p_value_wilcoxon": p_val_wilcoxon,
        "wilcoxon_note": (
            "For K=3, the minimum achievable two-sided Wilcoxon p-value is 0.25 "
            "due to finite permutation count (2^3 = 8). Formal rejection at p < 0.05 requires K >= 5."
            if K <= 4 else "K is sufficient for standard Wilcoxon rejection thresholds."
        ),
    }


def run_phase3_benchmark(
    profile: str = "canonical",
    output_dir: str = "experiments/phase3_canonical",
    phase2_ckpt_dir: str = "experiments/phase2_canonical_v2/models",
    force_retrain_noise: bool = False,
    num_workers: int = 1,
):
    """
    Executes the six-arm Phase 3 benchmark across all configured training replicates.
    """
    cfg: ExperimentConfig = get_config(profile)
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    replicates_dir = os.path.join(output_dir, "replicates")
    os.makedirs(replicates_dir, exist_ok=True)

    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]
    noise_scenarios = cfg.noise_scenarios
    training_seeds = cfg.training_seeds

    print("=" * 75)
    print(f"  MARLTSC Phase 3: Six-Arm Multi-Replicate Robustness Benchmark [{profile.upper()}]")
    print("=" * 75)
    print(f"  Grid Dimensions   : {cfg.num_intersections} intersections")
    print(f"  Simulation Steps  : {cfg.max_steps} steps/episode")
    print(f"  Training Seeds    : {training_seeds} (K={len(training_seeds)} replicates)")
    print(f"  Evaluation Seeds  : {cfg.num_seeds} base seeds x {cfg.eval_episodes_per_seed} eps")
    print(f"  Output Directory  : {output_dir}")
    print(f"  BSE Seeds         : Train={cfg.bse_train_seed}, Validation={cfg.bse_test_seed}")
    print("=" * 75)

    # -----------------------------------------------------------------
    # Step 1: Layer 1 GRU-KalmanNet Pre-Training (Phase A)
    # -----------------------------------------------------------------
    print("\n[Step 1/4] Layer 1 GRU-KalmanNet Supervised Pre-Training...")
    env_bse = SUMOEnv(
        net_file=cfg.net_file,
        route_file=cfg.route_file,
        num_intersections=cfg.num_intersections,
        max_steps=cfg.max_steps,
        seed=cfg.bse_train_seed,
    )

    bse_model = GRUKalmanNet(
        raw_obs_dim=16,
        hidden_dim=64,
        belief_dim=16,
        history_len=5,
    )

    bse_ckpt_path = os.path.join(model_dir, "bse.pt")
    if os.path.exists(bse_ckpt_path) and not force_retrain_noise:
        print(f"  -> Loading existing pre-trained BSE from {bse_ckpt_path}")
        data = torch.load(bse_ckpt_path, map_location="cpu", weights_only=False)
        bse_model.load_state_dict(data["bse_state"] if "bse_state" in data else data)
        X_val, Y_val = collect_bse_dataset(
            env_bse, num_steps=max(100, cfg.bse_steps // 4), seed=cfg.bse_test_seed
        )
        calib_metrics = evaluate_bse_calibration(bse_model, X_val, Y_val)
    else:
        print(f"  -> Collecting training pairs (seed={cfg.bse_train_seed}, steps={cfg.bse_steps})...")
        X_train, Y_train = collect_bse_dataset(
            env_bse, num_steps=cfg.bse_steps, seed=cfg.bse_train_seed
        )
        print(f"  -> Collecting held-out validation pairs (seed={cfg.bse_test_seed})...")
        X_val, Y_val = collect_bse_dataset(
            env_bse, num_steps=max(100, cfg.bse_steps // 4), seed=cfg.bse_test_seed
        )

        print(f"  -> Training GRU-KalmanNet for {cfg.bse_epochs} epochs...")
        train_bse_supervised(bse_model, X_train, Y_train, epochs=cfg.bse_epochs, batch_size=64)
        calib_metrics = evaluate_bse_calibration(bse_model, X_val, Y_val)

        torch.save({"bse_state": bse_model.state_dict(), "metrics": calib_metrics}, bse_ckpt_path)
        print(f"  -> BSE checkpoint saved -> {bse_ckpt_path}")

    print(f"  * Raw Obs MSE       : {calib_metrics['raw_mse']:.4f}")
    print(f"  * BSE Denoised MSE   : {calib_metrics['bse_mse']:.4f} ({calib_metrics['mse_reduction_pct']:.1f}% reduction)")
    print(f"  * Uncertainty Corr  : r = {calib_metrics['sigma_calibration_correlation']:.3f}")

    with open(os.path.join(output_dir, "bse_calibration_metrics.json"), "w") as f:
        json.dump(calib_metrics, f, indent=2)

    # -----------------------------------------------------------------
    # Step 2: Multi-Replicate Training & Evaluation
    # -----------------------------------------------------------------
    print("\n[Step 2/4] Executing Benchmark Across Training Replicates...")

    arm_names = [
        "M1: IPPO (Clean)",
        "M2: IPPO (Noise-Trained)",
        "M3: IPPO + BSE (Ours)",
        "M4: FMA2C (Clean)",
        "M5: FMA2C (Noise-Trained)",
        "M6: FMA2C + BSE (Ours)",
    ]

    all_replicate_results = {}

    def get_clean_controller(cls_name: str, seed: int):
        """Builds a clean-trained controller with proper weights for the requested seed."""
        if cls_name == "ippo":
            ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
            p2_file = f"ippo_s{seed}.pt" if seed != 42 else "ippo.pt"
            p2_path = os.path.join(phase2_ckpt_dir, p2_file)
            if os.path.exists(p2_path) and cfg.num_intersections == 16:
                ctrl.load_model(p2_path)
            else:
                env_tr = SUMOEnv(num_intersections=cfg.num_intersections, max_steps=cfg.max_steps, seed=seed)
                tr = Trainer(env=env_tr)
                for ep in range(cfg.train_episodes):
                    tr.train_episode(ctrl, max_steps=cfg.max_steps, seed=seed + ep)
            return ctrl
        elif cls_name == "fma2c":
            ctrl = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
            p2_file = f"fma2c_s{seed}.pt" if seed != 42 else "fma2c.pt"
            p2_path = os.path.join(phase2_ckpt_dir, p2_file)
            if os.path.exists(p2_path) and cfg.num_intersections == 16:
                ctrl.load_model(p2_path)
            else:
                env_tr = SUMOEnv(num_intersections=cfg.num_intersections, max_steps=cfg.max_steps, seed=seed)
                tr = Trainer(env=env_tr)
                for ep in range(cfg.train_episodes):
                    tr.train_episode(ctrl, max_steps=cfg.max_steps, seed=seed + ep)
            return ctrl

    for train_seed in training_seeds:
        print(f"\n" + "-" * 60)
        print(f"  --- Processing Training Replicate Seed: {train_seed} ---")
        print("-" * 60)

        # 1. Arm M1: Clean IPPO
        m1_ippo = get_clean_controller("ippo", seed=train_seed)

        # 2. Arm M2: Noise-Trained IPPO
        m2_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
        m2_ckpt = os.path.join(model_dir, f"ippo_noise_s{train_seed}.pt")
        if os.path.exists(m2_ckpt) and not force_retrain_noise:
            print(f"  -> Arm M2: Loaded trained noise-augmented IPPO (seed={train_seed}) from {m2_ckpt}")
            m2_ippo.load_model(m2_ckpt)
        else:
            print(f"  -> Arm M2: Training IPPO under multi-noise mixture (seed={train_seed})...")
            env_tr_noise = SUMOEnv(num_intersections=cfg.num_intersections, max_steps=cfg.max_steps, seed=train_seed)
            train_noise_augmented_controller(
                controller=m2_ippo,
                env=env_tr_noise,
                train_episodes=cfg.train_episodes,
                max_steps=cfg.max_steps,
                seed=train_seed,
                checkpoint_path=m2_ckpt,
                verbose=False,
            )

        # 3. Arm M3: Clean IPPO + BSE Wrapper (Distinct controller instance from M1)
        m3_ippo_base = get_clean_controller("ippo", seed=train_seed)
        m3_bse = BSEControllerWrapper(base_controller=m3_ippo_base, bse_model=bse_model, history_len=5)

        # 4. Arm M4: Clean FMA2C
        m4_fma2c = get_clean_controller("fma2c", seed=train_seed)

        # 5. Arm M5: Noise-Trained FMA2C
        m5_fma2c = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
        m5_ckpt = os.path.join(model_dir, f"fma2c_noise_s{train_seed}.pt")
        if os.path.exists(m5_ckpt) and not force_retrain_noise:
            print(f"  -> Arm M5: Loaded trained noise-augmented FMA2C (seed={train_seed}) from {m5_ckpt}")
            m5_fma2c.load_model(m5_ckpt)
        else:
            print(f"  -> Arm M5: Training FMA2C under multi-noise mixture (seed={train_seed})...")
            env_tr_noise = SUMOEnv(num_intersections=cfg.num_intersections, max_steps=cfg.max_steps, seed=train_seed)
            train_noise_augmented_controller(
                controller=m5_fma2c,
                env=env_tr_noise,
                train_episodes=cfg.train_episodes,
                max_steps=cfg.max_steps,
                seed=train_seed,
                checkpoint_path=m5_ckpt,
                verbose=False,
            )

        # 6. Arm M6: Clean FMA2C + BSE Wrapper (Distinct controller instance from M4)
        m6_fma2c_base = get_clean_controller("fma2c", seed=train_seed)
        m6_bse = BSEControllerWrapper(base_controller=m6_fma2c_base, bse_model=bse_model, history_len=5)

        arms = {
            "M1: IPPO (Clean)": m1_ippo,
            "M2: IPPO (Noise-Trained)": m2_ippo,
            "M3: IPPO + BSE (Ours)": m3_bse,
            "M4: FMA2C (Clean)": m4_fma2c,
            "M5: FMA2C (Noise-Trained)": m5_fma2c,
            "M6: FMA2C + BSE (Ours)": m6_bse,
        }

        # Checkpointed evaluation for this replicate
        rep_raw_path = os.path.join(replicates_dir, f"raw_results_s{train_seed}.json")
        checkpointer = ResultCheckpointer(rep_raw_path, scenarios=noise_scenarios, methods=list(arms.keys()))

        eval_base_seeds = cfg.eval_base_seeds[: cfg.num_seeds]
        eval_episodes_per_seed = cfg.eval_episodes_per_seed

        for noise in noise_scenarios:
            seed_idx = 0
            for base_seed in eval_base_seeds:
                for k in range(eval_episodes_per_seed):
                    ep_seed = base_seed + k
                    all_done = all(checkpointer.is_completed(noise, arm_name, seed_idx) for arm_name in arms.keys())
                    if all_done:
                        seed_idx += 1
                        continue

                    env_eval = SUMOEnv(
                        net_file=cfg.net_file,
                        route_file=cfg.route_file,
                        num_intersections=cfg.num_intersections,
                        noise_scenario=noise,
                        max_steps=cfg.max_steps,
                        seed=ep_seed,
                    )
                    trainer_eval = Trainer(env=env_eval)

                    for arm_name, ctrl in arms.items():
                        if checkpointer.is_completed(noise, arm_name, seed_idx):
                            continue

                        if hasattr(ctrl, "reset"):
                            ctrl.reset(agent_ids)

                        summary = trainer_eval.run_evaluation_episode(
                            ctrl, max_steps=cfg.max_steps, seed=ep_seed
                        )

                        bse_latency = 0.0
                        if isinstance(ctrl, BSEControllerWrapper):
                            bse_latency = ctrl.get_average_latency_ms()

                        checkpointer.record_result(
                            scenario=noise,
                            method=arm_name,
                            avg_delay=summary["avg_delay"],
                            violation_rate=summary["constraint_violation_rate"],
                            throughput=summary.get("total_throughput", 0.0),
                            bse_latency_ms=bse_latency,
                        )

                    seed_idx += 1

        all_replicate_results[str(train_seed)] = checkpointer.get_results()

    # -----------------------------------------------------------------
    # Step 3: Replicate-Level Statistical Aggregation & Analysis
    # -----------------------------------------------------------------
    print("\n[Step 3/4] Performing Replicate-Level Statistical Analysis across K Replicates...")

    aggregated_summary = {
        "scenarios": {},
        "paired_comparisons": {},
        "training_seeds": training_seeds,
        "k_replicates": len(training_seeds),
    }

    # Consolidated raw results structure
    consolidated_raw = {
        "replicates": all_replicate_results,
        "scenarios": {},
    }

    for noise in noise_scenarios:
        aggregated_summary["scenarios"][noise] = {}
        consolidated_raw["scenarios"][noise] = {}

        for arm_name in arm_names:
            replicate_means_delay = []
            replicate_means_viol = []

            for train_seed in training_seeds:
                rep_data = all_replicate_results[str(train_seed)][noise][arm_name]
                delays = rep_data.get("avg_delay", [])
                viols = rep_data.get("violation_rate", [])

                if delays:
                    replicate_means_delay.append(float(np.mean(delays)))
                if viols:
                    replicate_means_viol.append(float(np.mean(viols)))

            stat_delay = compute_replicate_statistics(replicate_means_delay)
            stat_viol = compute_replicate_statistics(replicate_means_viol)

            aggregated_summary["scenarios"][noise][arm_name] = {
                "queue_length": stat_delay,
                "violation_rate": stat_viol,
            }
            consolidated_raw["scenarios"][noise][arm_name] = {
                "queue_length_replicate_means": stat_delay["replicate_means"],
                "queue_length_mean": stat_delay["mean"],
                "queue_length_std": stat_delay["std"],
                "queue_length_ci95": stat_delay["ci_95"],
            }

            print(f"  [{noise.upper():8s}] {arm_name:28s} | Queue: {stat_delay['mean']:5.2f} +/- {stat_delay['std']:4.2f} veh | Starve: {stat_viol['mean']*100.0:4.1f}%")

    # Paired hypothesis testing across replicates for Clean and Severe scenarios
    for test_scenario in ["clean", "severe"]:
        aggregated_summary["paired_comparisons"][test_scenario] = {}
        scenario_data = aggregated_summary["scenarios"][test_scenario]

        comparisons = [
            ("M3 vs M1 (BSE vs Clean IPPO)", "M3: IPPO + BSE (Ours)", "M1: IPPO (Clean)"),
            ("M3 vs M2 (BSE vs Noise-Trained IPPO)", "M3: IPPO + BSE (Ours)", "M2: IPPO (Noise-Trained)"),
            ("M6 vs M4 (BSE vs Clean FMA2C)", "M6: FMA2C + BSE (Ours)", "M4: FMA2C (Clean)"),
            ("M6 vs M5 (BSE vs Noise-Trained FMA2C)", "M6: FMA2C + BSE (Ours)", "M5: FMA2C (Noise-Trained)"),
        ]

        for comp_name, treat_name, base_name in comparisons:
            t_means = scenario_data[treat_name]["queue_length"]["replicate_means"]
            b_means = scenario_data[base_name]["queue_length"]["replicate_means"]
            paired_res = compute_paired_comparison(t_means, b_means)
            aggregated_summary["paired_comparisons"][test_scenario][comp_name] = paired_res

    # Save consolidated raw results and statistical report
    with open(os.path.join(output_dir, "phase3_raw_results.json"), "w") as f:
        json.dump(consolidated_raw, f, indent=2)

    with open(os.path.join(output_dir, "phase3_statistical_summary.json"), "w") as f:
        json.dump(aggregated_summary, f, indent=2)

    # -----------------------------------------------------------------
    # Step 4: Figures & Cryptographic Manifest
    # -----------------------------------------------------------------
    print("\n[Step 4/4] Generating Figures & Cryptographic Manifest...")
    setup_plot_style()

    # Fig 1: Clean Performance Preservation
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    clean_means = [aggregated_summary["scenarios"]["clean"][m]["queue_length"]["mean"] for m in arm_names]
    clean_sems = [aggregated_summary["scenarios"]["clean"][m]["queue_length"]["sem"] for m in arm_names]
    bars = ax.bar(arm_names, clean_means, yerr=clean_sems, capsize=4, color=["#4c72b0", "#dd8452", "#55a868", "#8172b3", "#ccb974", "#64b5cd"], width=0.55, edgecolor="black")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.1, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Average Queue Length (veh)")
    ax.set_title("Clean Reference Performance Preservation (K Replicates)")
    plt.xticks(rotation=20, ha="right", fontsize=9)
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "fig1_clean_preservation.png")
    plt.savefig(fig1_path)
    plt.close()

    # Fig 2: Robustness Degradation Curves across Noise Sweep
    delay_sweeps = {
        m: [aggregated_summary["scenarios"][n][m]["queue_length"]["mean"] for n in noise_scenarios]
        for m in arm_names
    }
    fig2_path = os.path.join(output_dir, "fig2_robustness_noise_sweep.png")
    plot_robustness_degradation(noise_scenarios, delay_sweeps, output_path=fig2_path)

    # Fig 3: BSE vs Data Augmentation (M1 vs M2 vs M3)
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    sub_arms = ["M1: IPPO (Clean)", "M2: IPPO (Noise-Trained)", "M3: IPPO + BSE (Ours)"]
    x = np.arange(len(noise_scenarios))
    markers = ["o", "s", "^"]
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    for idx, sa in enumerate(sub_arms):
        ax.plot(x, delay_sweeps[sa], marker=markers[idx], linewidth=2.0, color=colors[idx], label=sa)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in noise_scenarios])
    ax.set_xlabel("Noise Scenario")
    ax.set_ylabel("Queue Length (veh)")
    ax.set_title("BSE vs. Noise Data Augmentation (IPPO Family)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "fig3_bse_vs_noise_training.png")
    plt.savefig(fig3_path)
    plt.close()

    # Fig 4: BSE Calibration & Quality
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=300)
    ax.bar(["Raw Obs MSE", "BSE Denoised MSE"], [calib_metrics["raw_mse"], calib_metrics["bse_mse"]], color=["#c44e52", "#55a868"], width=0.4, edgecolor="black")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title(f"BSE Denoising Quality ({calib_metrics['mse_reduction_pct']:.1f}% Reduction)")
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "fig4_bse_calibration_quality.png")
    plt.savefig(fig4_path)
    plt.close()

    # Build Cryptographic Manifest & Detached Checksum File
    print("  -> Computing cryptographic artifact checksums (SHA-256)...")
    git_meta = get_git_metadata()

    # Collect all generated artifacts
    tracked_artifacts = [
        os.path.join(output_dir, "phase3_raw_results.json"),
        os.path.join(output_dir, "bse_calibration_metrics.json"),
        os.path.join(output_dir, "phase3_statistical_summary.json"),
        fig1_path,
        fig2_path,
        fig3_path,
        fig4_path,
        bse_ckpt_path,
    ]
    # Add all model checkpoints in model_dir
    for f in os.listdir(model_dir):
        if f.endswith(".pt"):
            tracked_artifacts.append(os.path.join(model_dir, f))

    artifact_hashes = {}
    checksum_lines = []
    for art_path in sorted(set(tracked_artifacts)):
        if os.path.exists(art_path):
            rel_name = os.path.relpath(art_path, output_dir).replace("\\", "/")
            h = compute_sha256(art_path)
            artifact_hashes[rel_name] = h
            checksum_lines.append(f"{h}  {rel_name}\n")

    # Write detached checksums file
    checksum_file = os.path.join(output_dir, "phase3_checksums.sha256")
    with open(checksum_file, "w") as f:
        f.writelines(checksum_lines)

    # Configuration SHA-256 hash
    cfg_json = json.dumps(cfg.to_dict(), sort_keys=True)
    cfg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "profile": profile,
        "config": cfg.to_dict(),
        "config_sha256": cfg_hash,
        "git": git_meta,
        "status": "COMPLETED",
        "k_training_replicates": len(training_seeds),
        "training_seeds": training_seeds,
        "arms_evaluated": arm_names,
        "scenarios_evaluated": noise_scenarios,
        "calibration_summary": calib_metrics,
        "artifact_hashes": artifact_hashes,
        "detached_checksum_file": "phase3_checksums.sha256",
        "manifest_exclusion_note": "The manifest itself is excluded from artifact_hashes to prevent circular reference.",
    }

    manifest_path = os.path.join(output_dir, "phase3_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 75)
    print("  [SUCCESS] Six-Arm Multi-Replicate Phase 3 Benchmark Completed Successfully!")
    print(f"  Artifacts & Cryptographic Checksums Saved -> {output_dir}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Six-Arm Multi-Replicate Phase 3 Benchmark Runner")
    parser.add_argument("--profile", type=str, default="canonical", choices=["canonical", "fast", "smoke"],
                        help="Configuration profile to execute (canonical, fast, smoke)")
    parser.add_argument("--output-dir", type=str, default="experiments/phase3_canonical",
                        help="Directory to save experimental results, models, and figures")
    parser.add_argument("--phase2-ckpt-dir", type=str, default="experiments/phase2_canonical_v2/models",
                        help="Directory containing pre-trained Phase 2 canonical checkpoints")
    parser.add_argument("--force-retrain-noise", action="store_true",
                        help="Force retraining of noise-augmented baseline models")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of workers for evaluation")

    args = parser.parse_args()

    run_phase3_benchmark(
        profile=args.profile,
        output_dir=args.output_dir,
        phase2_ckpt_dir=args.phase2_ckpt_dir,
        force_retrain_noise=args.force_retrain_noise,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
