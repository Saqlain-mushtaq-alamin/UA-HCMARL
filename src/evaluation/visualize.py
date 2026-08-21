"""
Visualization Utilities (Phase 1 / Phase 2 §2.5)

Plots:
- Delay vs Noise Level (Robustness degradation curves)
- Metric comparisons with 95% CIs
- Constraint violation breakdowns (MAPPO-LCE vs Unconstrained under noise)
- FMA2C Fragility Diagnostic (Hierarchy amplifies noise sensitivity)
"""

import os
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def setup_plot_style():
    """Applies clean publication-quality styling."""
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 13
    plt.rcParams["xtick.labelsize"] = 10
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 10
    plt.rcParams["figure.titlesize"] = 14


def plot_robustness_degradation(
    noise_scenarios: List[str],
    results_by_method: Dict[str, List[float]],
    output_path: str = "experiments/robustness_degradation.png",
):
    """Plots Average Delay across Noise Degradation Levels for different methods."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    x_indices = np.arange(len(noise_scenarios))

    for method_name, delays in results_by_method.items():
        ax.plot(x_indices, delays, marker="o", linewidth=2.0, label=method_name)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([s.capitalize() for s in noise_scenarios])
    ax.set_xlabel("Noise Degradation Scenario (N5 Protocol)")
    ax.set_ylabel("Average Vehicle Delay (seconds)")
    ax.set_title("Robustness under Sensor Noise & Degradation")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_baseline_comparison(
    metrics_summary: Dict[str, Dict[str, float]],
    metric_name: str = "avg_delay",
    output_path: str = "experiments/baseline_comparison.png",
):
    """Bar chart comparing baselines on a given metric."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    methods = list(metrics_summary.keys())
    values = [metrics_summary[m].get(metric_name, 0.0) for m in methods]

    palette = sns.color_palette("muted", len(methods))
    bars = ax.bar(methods, values, color=palette, width=0.5, edgecolor="black")

    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + 0.5,
            f"{yval:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(f"Clean Baseline Comparison: {metric_name.replace('_', ' ').title()}")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_constraint_violations(
    noise_scenarios: List[str],
    violations_by_method: Dict[str, List[float]],
    output_path: str = "experiments/constraint_violations_noise.png",
):
    """Plots Constraint Violation Rates across Noise Degradation Levels (Phase 2 Milestone Fig 4)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    x_indices = np.arange(len(noise_scenarios))

    for method_name, violation_rates in violations_by_method.items():
        ax.plot(x_indices, [v * 100.0 for v in violation_rates], marker="s", linewidth=2.0, label=method_name)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([s.capitalize() for s in noise_scenarios])
    ax.set_xlabel("Noise Degradation Scenario (N5 Protocol)")
    ax.set_ylabel("Constraint Violation Rate (%)")
    ax.set_title("Constraint Violation Rate under Sensor Noise Sweep")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_hierarchy_fragility_diagnostic(
    noise_scenarios: List[str],
    delays_by_method: Dict[str, List[float]],
    output_path: str = "experiments/fma2c_fragility_diagnostic.png",
):
    """Plots FMA2C Fragility Diagnostic vs MAPPO/IPPO under Noise (Phase 2 Milestone Fig 5 / Novelty N4)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    x_indices = np.arange(len(noise_scenarios))

    for method_name, delays in delays_by_method.items():
        style = "--" if "FMA2C" in method_name else "-"
        marker = "^" if "FMA2C" in method_name else "o"
        ax.plot(x_indices, delays, linestyle=style, marker=marker, linewidth=2.2, label=method_name)

    ax.set_xticks(x_indices)
    ax.set_xticklabels([s.capitalize() for s in noise_scenarios])
    ax.set_xlabel("Noise Degradation Scenario (N5 Protocol)")
    ax.set_ylabel("Average Delay (s)")
    ax.set_title("FMA2C Fragility Diagnostic: Unconstrained Hierarchy Noise Sensitivity")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
