"""Generate all publication figures from existing experiment JSON files."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

NOISE_SCENARIOS = ["clean", "mild", "moderate", "severe", "extreme"]
NOISE_LEVELS    = [0.0, 0.1, 0.2, 0.3, 0.5]

COLORS = {
    "Fixed-Time":            "#555555",
    "Actuated":              "#888888",
    "IPPO":                  "#2196F3",
    "MAPPO":                 "#4CAF50",
    "MAPPO-LCE":             "#FF9800",
    "FMA2C":                 "#9C27B0",
    "MAPPO-LCE (Vanilla)":   "#FF9800",
    "BSE+MAPPO-LCE (No-Mod)":"#F44336",
    "BSE+MAPPO-LCE (Proposed)":"#00BCD4",
    "BSE+MAPPO-LCE":         "#00BCD4",
    "UA-HCMARL":             "#E91E63",
}

def style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

def load(path):
    with open(path) as f:
        return json.load(f)

def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure -> {path}")

style()

# -------------------------------------------------------
# FIGURE 1: Phase 2 Clean Baseline Comparison (bar chart)
# -------------------------------------------------------
p2 = load("experiments/phase2_flat_baselines/phase2_raw_results.json")
methods_p2 = list(p2["clean"].keys())
clean_means = [np.mean(p2["clean"][m]["avg_delay"]) for m in methods_p2]
clean_stds  = [np.std(p2["clean"][m]["avg_delay"])  for m in methods_p2]
colors_p2   = [COLORS.get(m, "#607D8B") for m in methods_p2]

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(methods_p2))
bars = ax.bar(x, clean_means, yerr=clean_stds, capsize=4,
              color=colors_p2, edgecolor="white", linewidth=0.8, alpha=0.9)
for bar, val in zip(bars, clean_means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(methods_p2, rotation=15, ha="right")
ax.set_ylabel("Average Vehicle Delay (s)")
ax.set_title("Figure 1: Clean Condition — Flat Baseline Comparison\n(RL methods vs. Fixed-Time & Actuated controllers)")
ax.axhline(clean_means[0], color=COLORS["Fixed-Time"], linestyle="--", linewidth=1.2, label="Fixed-Time level")
ax.legend()
savefig("experiments/phase2_flat_baselines/fig1_clean_baseline_comparison.png")

# -------------------------------------------------------
# FIGURE 2: Robustness Degradation Curves
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for m in methods_p2:
    delays = [np.mean(p2[n][m]["avg_delay"]) for n in NOISE_SCENARIOS]
    stds   = [np.std(p2[n][m]["avg_delay"])  for n in NOISE_SCENARIOS]
    color  = COLORS.get(m, "#607D8B")
    ls     = "--" if m in ["Fixed-Time", "Actuated"] else "-"
    marker = "s" if m in ["Fixed-Time", "Actuated"] else "o"
    ax.plot(NOISE_LEVELS, delays, color=color, linestyle=ls, marker=marker,
            linewidth=2, markersize=6, label=m)
    ax.fill_between(NOISE_LEVELS,
                    [d - s for d, s in zip(delays, stds)],
                    [d + s for d, s in zip(delays, stds)],
                    color=color, alpha=0.1)
ax.set_xlabel("Noise Level (sigma_D1)")
ax.set_ylabel("Average Vehicle Delay (s)")
ax.set_title("Figure 2: Robustness under Sensor Noise Degradation (N5 Protocol)")
ax.set_xticks(NOISE_LEVELS)
ax.set_xticklabels(NOISE_SCENARIOS)
ax.legend(loc="upper left", ncol=2)
savefig("experiments/phase2_flat_baselines/fig2_robustness_degradation.png")

# -------------------------------------------------------
# FIGURE 3: Constraint Violation Rates vs Noise
# -------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for m in methods_p2:
    viols = [np.mean(p2[n][m]["violation_rate"]) * 100 for n in NOISE_SCENARIOS]
    ax.plot(NOISE_LEVELS, viols, color=COLORS.get(m, "#607D8B"),
            marker="^", linewidth=2, markersize=6, label=m)
ax.set_xlabel("Noise Level")
ax.set_ylabel("Constraint Violation Rate (%)")
ax.set_title("Figure 3: Constraint Violation Rate vs. Noise Level")
ax.set_xticks(NOISE_LEVELS)
ax.set_xticklabels(NOISE_SCENARIOS)
ax.legend(ncol=2)
savefig("experiments/phase2_flat_baselines/fig3_violation_rate_vs_noise.png")

# -------------------------------------------------------
# FIGURE 4: Statistical Summary — Effect Sizes
# -------------------------------------------------------
stat2 = load("experiments/phase2_flat_baselines/phase2_statistical_summary.json")
rl_methods = [m for m in methods_p2 if m not in ["Fixed-Time", "Actuated"]]
cohens_ds  = [stat2.get(m, {}).get("vs_baseline", {}).get("cohens_d", 0.0) for m in rl_methods]
p_values   = [stat2.get(m, {}).get("vs_baseline", {}).get("p_value", 1.0) for m in rl_methods]
colors_stat = [COLORS.get(m, "#607D8B") for m in rl_methods]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
bars = ax1.bar(rl_methods, cohens_ds, color=colors_stat, edgecolor="white", alpha=0.9)
ax1.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="Large effect (d=0.8)")
ax1.set_ylabel("Cohen's d (vs Fixed-Time)")
ax1.set_title("Effect Size: RL vs. Fixed-Time")
ax1.set_xticklabels(rl_methods, rotation=15, ha="right")
ax1.legend()

bars2 = ax2.bar(rl_methods, [-np.log10(max(p, 1e-10)) for p in p_values],
                color=colors_stat, edgecolor="white", alpha=0.9)
ax2.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.5, label="p=0.05 threshold")
ax2.set_ylabel("-log10(p-value)")
ax2.set_title("Statistical Significance (-log10 p)")
ax2.set_xticklabels(rl_methods, rotation=15, ha="right")
ax2.legend()
savefig("experiments/phase2_flat_baselines/fig4_statistical_significance.png")

# -------------------------------------------------------
# FIGURE 5 (Phase 5): Ablation Comparison
# -------------------------------------------------------
if os.path.exists("experiments/phase5_ablations/phase5_ablation_results.json"):
    p5 = load("experiments/phase5_ablations/phase5_ablation_results.json")
    abl_methods = list(p5.keys())
    abl_delays  = [p5[m]["mean_delay"] for m in abl_methods]
    abl_stds    = [np.std(p5[m]["avg_delay"]) for m in abl_methods]
    abl_colors  = []
    for m in abl_methods:
        if "UA-HCMARL" in m:    abl_colors.append(COLORS["UA-HCMARL"])
        elif "BSE+MAPPO" in m:  abl_colors.append(COLORS["BSE+MAPPO-LCE"])
        elif "MAPPO-LCE" in m:  abl_colors.append(COLORS["MAPPO-LCE"])
        elif "MAPPO" in m:      abl_colors.append(COLORS["MAPPO"])
        elif "FMA2C" in m:      abl_colors.append(COLORS["FMA2C"])
        else:                   abl_colors.append(COLORS["Fixed-Time"])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(abl_methods))
    bars = ax.bar(x, abl_delays, yerr=abl_stds, capsize=4,
                  color=abl_colors, edgecolor="white", alpha=0.9)
    for bar, val in zip(bars, abl_delays):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(":")[1].strip() if ":" in m else m for m in abl_methods],
                       rotation=20, ha="right")
    ax.set_ylabel("Average Vehicle Delay (s)")
    ax.set_title("Figure 5: Phase 5 Controlled Ablation — Clean Condition")
    savefig("experiments/phase5_ablations/fig5_ablation_comparison.png")

# -------------------------------------------------------
# FIGURE 6 (Phase 3): BSE Calibration Quality
# -------------------------------------------------------
if os.path.exists("experiments/phase3_belief_encoder/bse_calibration_metrics.json"):
    calib = load("experiments/phase3_belief_encoder/bse_calibration_metrics.json")
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(["Raw Obs MSE", "BSE Denoised MSE"],
                  [calib["raw_mse"], calib["bse_mse"]],
                  color=["#F44336", "#4CAF50"], edgecolor="white", width=0.45, alpha=0.9)
    for bar, val in zip(bars, [calib["raw_mse"], calib["bse_mse"]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Squared Error (MSE)")
    ax.set_title(f"Figure 6: BSE Denoising Quality\n"
                 f"({calib['mse_reduction_pct']:.1f}% MSE reduction, "
                 f"sigma-calibration r={calib['sigma_calibration_correlation']:.3f})")
    savefig("experiments/phase3_belief_encoder/fig6_bse_calibration.png")

# -------------------------------------------------------
# FIGURE 7: Phase 4 — Full System Noise Robustness
# -------------------------------------------------------
if os.path.exists("experiments/phase4_full_hierarchy/phase4_raw_results.json"):
    p4 = load("experiments/phase4_full_hierarchy/phase4_raw_results.json")
    p4_methods = list(p4["clean"].keys())
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in p4_methods:
        delays = [np.mean(p4[n][m]["avg_delay"]) for n in NOISE_SCENARIOS]
        color  = COLORS.get(m, "#607D8B")
        ls     = "--" if "Fixed-Time" in m else "-"
        marker = "s" if "Fixed-Time" in m else ("^" if "UA-HCMARL" in m else "o")
        lw     = 2.5 if "UA-HCMARL" in m else 2
        ax.plot(NOISE_LEVELS, delays, color=color, linestyle=ls,
                marker=marker, linewidth=lw, markersize=7, label=m)
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Average Vehicle Delay (s)")
    ax.set_title("Figure 7: Phase 4 — Full UA-HCMARL vs. Baselines (Noise Robustness)")
    ax.set_xticks(NOISE_LEVELS)
    ax.set_xticklabels(NOISE_SCENARIOS)
    ax.legend()
    savefig("experiments/phase4_full_hierarchy/fig7_ua_hcmarl_robustness.png")

print("\nAll figures generated successfully.")
