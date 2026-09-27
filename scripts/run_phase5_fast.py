"""
Phase 5 Ablation Runner (Config-driven & Checkpointed)
======================================================
Imports configuration from experiment_config.py to ensure standardized parameters.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from experiment_config import get_config
from scripts.run_phase5_ablation_benchmark import run_phase5_ablation_benchmark

cfg = get_config("canonical")

run_phase5_ablation_benchmark(
    net_file=cfg.net_file,
    route_file=cfg.route_file,
    num_intersections=cfg.num_intersections,
    num_seeds=cfg.num_seeds,
    max_steps=cfg.max_steps,
    output_dir="experiments/phase5_ablations",
)

