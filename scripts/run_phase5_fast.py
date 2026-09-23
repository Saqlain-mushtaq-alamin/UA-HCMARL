"""
Fast Phase 5 runner: num_seeds=10, max_steps=500, train_eps=5.
Produces valid Wilcoxon statistics. Overrides defaults without touching
the main benchmark script.
"""
import sys, os
sys.path.insert(0, os.path.abspath("."))

from scripts.run_phase5_ablation_benchmark import run_phase5_ablation_benchmark

run_phase5_ablation_benchmark(
    num_seeds=10,
    max_steps=500,
    output_dir="experiments/phase5_ablations",
)
