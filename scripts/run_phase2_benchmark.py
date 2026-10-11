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

High Performance Multi-Core Architecture:
- Batched forward passes across agents (16x fewer PyTorch graph dispatches)
- Multi-process parallel training (--parallel-train): Trains IPPO, MAPPO, MAPPO-LCE,
  and FMA2C simultaneously across 4 CPU cores
- Multi-process parallel evaluation (--num-workers): Parallelizes seed/scenario evaluation
  across up to 12-24 CPU cores
- Persistent model checkpoints: Saves and loads weights to experiments/.../models/
- Checkpoint validation: Supports --force-eval to avoid false-cache skips from prior runs
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import sys
import time
from typing import Dict, List, Any, Optional
import numpy as np
import torch

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
from experiment_config import get_config


def _reset_lagrangian_state(controller):
    """Reset Lagrangian multiplier state between episodes (prevents carry-over)."""
    if hasattr(controller, "pid_lagrangian"):
        controller.pid_lagrangian.reset()
    if hasattr(controller, "pid_lagrangians"):
        for lag in controller.pid_lagrangians.values():
            lag.reset()


def _create_controller(m_name: str, agent_ids: List[str], num_intersections: int, device: str = "cpu"):
    """Instantiate a controller by name."""
    if m_name == "Fixed-Time":
        return FixedTimeController()
    elif m_name == "Actuated":
        return ActuatedController()
    elif m_name == "IPPO":
        return IPPOController(agent_ids=agent_ids, device=device)
    elif m_name == "MAPPO":
        return MAPPOController(agent_ids=agent_ids, device=device)
    elif m_name == "MAPPO-LCE":
        return MAPPOLCEController(agent_ids=agent_ids, device=device)
    elif m_name == "FMA2C":
        return FMA2CController(
            agent_ids=agent_ids,
            num_regions=4 if num_intersections >= 4 else 2,
            device=device,
        )
    else:
        raise ValueError(f"Unknown controller: {m_name}")


def _train_single_baseline_worker(task_args):
    """Worker function for concurrent multi-process training."""
    m_name, net_file, route_file, num_intersections, max_steps, train_episodes, output_dir, device = task_args
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    env_train = SUMOEnv(
        net_file=net_file,
        route_file=route_file,
        num_intersections=num_intersections,
        demand_profile="uniform",
        noise_scenario="clean",
        max_steps=max_steps,
        seed=42,
    )
    logger = LearningCurveLogger()
    trainer_train = Trainer(env=env_train, learning_curve_logger=logger)
    ctrl = _create_controller(m_name, agent_ids, num_intersections, device=device)

    t0 = time.time()
    print(f"  [Worker PID {os.getpid()}] Starting training for {m_name} ({train_episodes} eps)...")
    best_delay = float("inf")
    for ep in range(train_episodes):
        ep_seed = 42 + ep
        summary, loss = trainer_train.train_episode(ctrl, max_steps=max_steps, seed=ep_seed)
        _reset_lagrangian_state(ctrl)
        if (ep + 1) % 50 == 0 or (ep + 1) == train_episodes:
            d = summary.get("avg_delay", float("nan"))
            elapsed = time.time() - t0
            print(f"     [{m_name}] ep {ep+1:4d}/{train_episodes} | delay={d:.2f} s | elapsed={elapsed/60.0:.1f}m")
            best_delay = min(best_delay, d)

    curves_path = os.path.join(output_dir, f"learning_curves_{m_name.replace('-', '_').lower()}.json")
    trainer_train.save_learning_curves(curves_path)

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{m_name.replace('-', '_').lower()}.pt")
    if hasattr(ctrl, "save_model"):
        ctrl.save_model(model_path)

    total_time = (time.time() - t0) / 60.0
    print(f"  [Worker PID {os.getpid()}] Finished {m_name} in {total_time:.1f}m! Best delay: {best_delay:.2f} s")
    return m_name, best_delay, model_path


def _eval_single_task_worker(task_args):
    """Worker function for concurrent multi-process evaluation."""
    noise, seed, noise_index, m_name, net_file, route_file, num_intersections, max_steps, eval_episodes_per_seed, model_path, device = task_args
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    env_seed = seed + 1000 + noise_index * 10000

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
    ctrl = _create_controller(m_name, agent_ids, num_intersections, device=device)
    if model_path and os.path.exists(model_path) and hasattr(ctrl, "load_model"):
        ctrl.load_model(model_path)

    seed_delays = []
    seed_viols = []
    for eval_ep in range(eval_episodes_per_seed):
        _reset_lagrangian_state(ctrl)
        ep_seed = seed + 1000 + eval_ep * 100 + noise_index * 10000
        ep_summary = trainer_eval.run_evaluation_episode(
            ctrl, max_steps=max_steps, seed=ep_seed
        )
        seed_delays.append(ep_summary["avg_delay"])
        seed_viols.append(ep_summary["constraint_violation_rate"])

    avg_d = float(np.mean(seed_delays))
    avg_v = float(np.mean(seed_viols))
    return noise, seed, m_name, avg_d, avg_v


def run_phase2_benchmark(
    net_file: str = "networks/grid4x4/grid4x4.net.xml",
    route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
    num_intersections: int = 16,
    num_seeds: int = 10,
    train_episodes: int = 200,
    eval_episodes_per_seed: int = 3,
    max_steps: int = 3600,
    output_dir: str = "experiments/phase2_flat_baselines",
    num_workers: Optional[int] = None,
    parallel_train: bool = True,
    force_eval: bool = False,
    skip_train: bool = False,
    force_train: bool = False,
    device: str = "cpu",
):
    """
    Full Phase 2 benchmark with multi-core parallelism and batched vectorization.
    """
    os.makedirs(output_dir, exist_ok=True)
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    all_methods = ["Fixed-Time", "Actuated", "IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]
    rl_methods = ["IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]

    cpu_count = os.cpu_count() or 4
    if num_workers is None:
        num_workers = min(12, cpu_count)

    print("==========================================================")
    print("   UA-HCMARL Phase 2: Flat Baseline Benchmark Suite       ")
    print("==========================================================")
    print(f"  CPU Cores Available: {cpu_count} | Workers Allocated: {num_workers}")
    print(f"  Config: {num_seeds} seeds | {train_episodes} train eps | {max_steps} steps/ep")
    print(f"  Parallel Training: {parallel_train} | Force Re-Eval: {force_eval}")
    print("==========================================================")

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_paths = {
        m: os.path.join(model_dir, f"{m.replace('-', '_').lower()}.pt") for m in rl_methods
    }

    # ------------------------------------------------------------------
    # 1. Train RL controllers on Clean condition
    # ------------------------------------------------------------------
    if skip_train:
        print("\n[Step 1/3] Skipping RL Training (--skip-train enabled). Using saved models.")
    else:
        print("\n[Step 1/3] Training RL Baseline Controllers (Clean Condition)...")
        tasks = []
        for m in rl_methods:
            curves_path = os.path.join(output_dir, f"learning_curves_{m.replace('-', '_').lower()}.json")
            m_path = model_paths[m]
            if not force_train and os.path.exists(m_path) and os.path.exists(curves_path):
                print(f"  [CACHE] {m} checkpoint already exists ({m_path}). Skipping re-training.")
            else:
                tasks.append(
                    (m, net_file, route_file, num_intersections, max_steps, train_episodes, output_dir, device)
                )

        if len(tasks) == 0:
            print("  All RL baseline models are already trained and cached. Proceeding directly to evaluation.")
        elif parallel_train and num_workers > 1:
            train_workers = min(len(tasks), num_workers)
            print(f"  -> Spawning {train_workers} parallel worker processes for {len(tasks)} model(s)...")
            with ProcessPoolExecutor(max_workers=train_workers) as executor:
                futures = {executor.submit(_train_single_baseline_worker, task): task[0] for task in tasks}
                for future in as_completed(futures):
                    m_name = futures[future]
                    try:
                        _, best_d, m_path = future.result()
                        print(f"  [OK] {m_name} trained successfully (best delay: {best_d:.2f} s).")
                    except Exception as exc:
                        print(f"  [ERROR] Training failed for {m_name}: {exc}")
                        raise
        else:
            # Sequential training fallback
            for task in tasks:
                _train_single_baseline_worker(task)

    # ------------------------------------------------------------------
    # 2. Evaluate across Seeds & Noise Sweeps
    # ------------------------------------------------------------------
    print("\n[Step 2/3] Evaluating All Baselines across Seeds & Noise Protocol...")
    raw_path = os.path.join(output_dir, "phase2_raw_results.json")

    # If force_eval is requested, backup or reset old results to avoid stale checkpoint bypass
    if force_eval and os.path.exists(raw_path):
        backup_path = raw_path.replace(".json", f"_backup_{int(time.time())}.json")
        try:
            os.rename(raw_path, backup_path)
            print(f"  [Notice] Backed up prior checkpoint to {backup_path} for fresh evaluation.")
        except OSError:
            pass

    checkpointer = ResultCheckpointer(raw_path, scenarios=noise_scenarios, methods=all_methods)

    # Build missing evaluation tasks
    eval_tasks = []
    for noise in noise_scenarios:
        noise_index = noise_scenarios.index(noise)
        for seed in range(num_seeds):
            for m_name in all_methods:
                if not force_eval and checkpointer.is_completed(noise, m_name, seed):
                    continue
                m_path = model_paths.get(m_name, None)
                eval_tasks.append((
                    noise, seed, noise_index, m_name, net_file, route_file,
                    num_intersections, max_steps, eval_episodes_per_seed, m_path, device
                ))

    total_eval_tasks = len(eval_tasks)
    if total_eval_tasks == 0:
        print("  All evaluation runs are already completed according to checkpoint.")
    else:
        print(f"  Executing {total_eval_tasks} evaluation tasks using {num_workers} parallel workers...")
        completed_count = 0
        t_eval_start = time.time()

        if num_workers > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                future_map = {executor.submit(_eval_single_task_worker, task): task for task in eval_tasks}
                for future in as_completed(future_map):
                    noise, seed, m_name, avg_d, avg_v = future.result()
                    checkpointer.record_result(
                        scenario=noise,
                        method=m_name,
                        avg_delay=avg_d,
                        violation_rate=avg_v,
                    )
                    completed_count += 1
                    if completed_count % 5 == 0 or completed_count == total_eval_tasks:
                        rate = completed_count / max(1.0, time.time() - t_eval_start)
                        rem_sec = (total_eval_tasks - completed_count) / max(0.01, rate)
                        print(f"    Eval Progress: {completed_count}/{total_eval_tasks} done "
                              f"({completed_count*100/total_eval_tasks:.1f}%) | ETA: {rem_sec/60.0:.1f}m | "
                              f"Last: [{noise.upper()}] {m_name} seed={seed} (delay={avg_d:.2f}s)")
        else:
            for task in eval_tasks:
                noise, seed, m_name, avg_d, avg_v = _eval_single_task_worker(task)
                checkpointer.record_result(
                    scenario=noise,
                    method=m_name,
                    avg_delay=avg_d,
                    violation_rate=avg_v,
                )
                completed_count += 1
                print(f"    [{noise.upper()}] {m_name:12s} seed={seed:2d} | delay={avg_d:.2f} s | viol={avg_v*100.0:.1f}%")

    # ------------------------------------------------------------------
    # 3. Print Results Summary
    # ------------------------------------------------------------------
    benchmark_results = checkpointer.get_results()
    print("\n--- Phase 2 Evaluation Summary ---")
    for noise in noise_scenarios:
        print(f"\n  Scenario: [{noise.upper()}]")
        for m_name in all_methods:
            delays = benchmark_results.get(noise, {}).get(m_name, {}).get("avg_delay", [])
            viols = benchmark_results.get(noise, {}).get(m_name, {}).get("violation_rate", [])
            m_delay = float(np.mean(delays)) if len(delays) > 0 else float("nan")
            m_viol = float(np.mean(viols)) if len(viols) > 0 else float("nan")
            print(f"    {m_name:12s} | Delay: {m_delay:6.2f} s | Violation Rate: {m_viol*100.0:5.1f}%")

    with open(raw_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\n[Success] Phase 2 raw benchmark data saved to {raw_path}")

    # ------------------------------------------------------------------
    # 4. Statistical Report (paired t-test + Wilcoxon)
    # ------------------------------------------------------------------
    print("\n[Step 3/3] Generating Statistical Significance & Robustness Reports...")
    clean_delays = {m: benchmark_results["clean"][m]["avg_delay"] for m in all_methods}
    stat_report = generate_statistical_report(clean_delays, baseline_name="Fixed-Time")

    # Augment with paired t-test
    fixed_arr = np.array(clean_delays["Fixed-Time"])
    for m_name in all_methods:
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
    for m_name in all_methods:
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
    for m_name in all_methods:
        delays_over_noise = [
            float(np.mean(benchmark_results[n][m_name]["avg_delay"])) for n in noise_scenarios
        ]
        slope, intercept, r_val = compute_robustness_degradation_slope(noise_indices, delays_over_noise)
        robustness_slopes[m_name] = {"slope": slope, "r_value": r_val, "delays": delays_over_noise}
        sign = "[Y] OK" if slope >= 0 else "[N] INVERTED"
        print(f"  {m_name:12s} slope={slope:+8.2f}  r={r_val:.2f}  {sign}")

        if m_name == "Fixed-Time":
            if len(set(round(d, 4) for d in delays_over_noise)) == 1:
                print("  [ERROR] Fixed-Time delays are IDENTICAL across noise levels!")
            else:
                print("  [OK] Fixed-Time delays vary across noise levels")

    slopes_path = os.path.join(output_dir, "robustness_slopes.json")
    with open(slopes_path, "w") as f:
        json.dump(robustness_slopes, f, indent=2)

    # ------------------------------------------------------------------
    # 6. Generate Milestone Figures
    # ------------------------------------------------------------------
    clean_summary = {m: {"avg_delay": float(np.mean(clean_delays[m]))} for m in all_methods}
    plot_baseline_comparison(
        clean_summary, metric_name="avg_delay",
        output_path=os.path.join(output_dir, "fig1_clean_baseline_comparison.png"),
    )

    robustness_delays = {
        m: [float(np.mean(benchmark_results[n][m]["avg_delay"])) for n in noise_scenarios]
        for m in all_methods
    }
    plot_robustness_degradation(
        noise_scenarios, robustness_delays,
        output_path=os.path.join(output_dir, "fig2_robustness_degradation.png"),
    )

    violation_rates = {
        m: [float(np.mean(benchmark_results[n][m]["violation_rate"])) for n in noise_scenarios]
        for m in all_methods
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


def main():
    parser = argparse.ArgumentParser(description="Phase 2 High-Performance Benchmark Runner")
    parser.add_argument("--profile", type=str, default=None, choices=["canonical", "fast", "smoke"],
                        help="Configuration profile to execute (canonical, fast, smoke)")
    parser.add_argument("--num-workers", type=int, default=min(12, os.cpu_count() or 4),
                        help="Number of parallel worker processes for evaluation and training")
    parser.add_argument("--parallel-train", action="store_true", default=True,
                        help="Train RL baselines concurrently across CPU cores")
    parser.add_argument("--no-parallel-train", dest="parallel_train", action="store_false",
                        help="Train RL baselines sequentially")
    parser.add_argument("--force-eval", action="store_true", default=False,
                        help="Force fresh evaluation across all seeds/noise, overriding previous checkpoint")
    parser.add_argument("--skip-train", action="store_true", default=False,
                        help="Skip training and load existing model checkpoints from disk")
    parser.add_argument("--force-train", action="store_true", default=False,
                        help="Force re-training of RL models even if cached checkpoints exist")
    parser.add_argument("--num-seeds", type=int, default=10,
                        help="Number of random seeds for evaluation")
    parser.add_argument("--train-episodes", type=int, default=200,
                        help="Number of training episodes per RL controller")
    parser.add_argument("--eval-episodes", type=int, default=3,
                        help="Number of evaluation episodes per seed")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="Number of simulation steps per episode")
    parser.add_argument("--device", type=str, default="cpu",
                        help="PyTorch device ('cpu' or 'cuda')")
    parser.add_argument("--output-dir", type=str, default="experiments/phase2_flat_baselines",
                        help="Output directory for results and figures")

    args = parser.parse_args()

    net_file = "networks/grid4x4/grid4x4.net.xml"
    route_file = "networks/grid4x4/demand_uniform.rou.xml"
    num_intersections = 16
    num_seeds = args.num_seeds
    train_episodes = args.train_episodes
    eval_episodes = args.eval_episodes
    max_steps = args.max_steps

    if args.profile:
        cfg = get_config(args.profile)
        net_file = cfg.net_file
        route_file = cfg.route_file
        num_intersections = cfg.num_intersections
        if args.num_seeds == parser.get_default("num_seeds"):
            num_seeds = cfg.num_seeds
        if args.train_episodes == parser.get_default("train_episodes"):
            train_episodes = cfg.train_episodes
        if args.eval_episodes == parser.get_default("eval_episodes"):
            eval_episodes = cfg.eval_episodes_per_seed
        if args.max_steps == parser.get_default("max_steps"):
            max_steps = cfg.max_steps

    run_phase2_benchmark(
        net_file=net_file,
        route_file=route_file,
        num_intersections=num_intersections,
        num_seeds=num_seeds,
        train_episodes=train_episodes,
        eval_episodes_per_seed=eval_episodes,
        max_steps=max_steps,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        parallel_train=args.parallel_train,
        force_eval=args.force_eval,
        skip_train=args.skip_train,
        force_train=args.force_train,
        device=args.device,
    )


if __name__ == "__main__":
    main()
