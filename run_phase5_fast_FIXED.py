"""
run_phase5_fast_FIXED.py — Canonical & Resumable Phase 5 Ablation Runner
========================================================================
Supports:
  python run_phase5_fast_FIXED.py --profile canonical   # Full canonical publication run (3600 steps, 10 seeds)
  python run_phase5_fast_FIXED.py --profile fast        # Quick run (500 steps, 5 seeds)
  python run_phase5_fast_FIXED.py --profile smoke       # Rapid sanity test (100 steps, 2 seeds)
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath("."))

from experiment_config import get_config
from scripts.run_phase5_ablation_benchmark import run_phase5_ablation_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Ablation Benchmark with Config Profile")
    parser.add_argument(
        "--profile",
        type=str,
        default="canonical",
        choices=["canonical", "fast", "smoke"],
        help="Configuration profile to execute (default: canonical)",
    )
    args = parser.parse_args()

    cfg = get_config(args.profile)
    print("=" * 65)
    print(f"  UA-HCMARL Phase 5 Ablation Benchmark [Profile: {cfg.profile_name.upper()}]")
    print("=" * 65)
    print(f"  Network:            {cfg.net_file} ({cfg.num_intersections} intersections)")
    print(f"  Horizon (max_steps): {cfg.max_steps} s per episode")
    print(f"  Seeds:              {cfg.num_seeds} seeds")
    print(f"  Noise Scenarios:    {cfg.noise_scenarios}")
    print("=" * 65)

    run_phase5_ablation_benchmark(
        net_file=cfg.net_file,
        route_file=cfg.route_file,
        num_intersections=cfg.num_intersections,
        num_seeds=cfg.num_seeds,
        max_steps=cfg.max_steps,
        output_dir="experiments/phase5_ablations",
    )


if __name__ == "__main__":
    main()
