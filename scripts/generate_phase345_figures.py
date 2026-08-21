"""Generate Phase 3 BSE-specific figures and updated Phase 5 ablation figure."""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

NOISE_SCENARIOS = ["clean", "mild", "moderate", "severe", "extreme"]
NOISE_LEVELS    = [0.0, 0.1, 0.2, 0.3, 0.5]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "figure.dpi": 150,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

def load(path):
    with open(path) as f: return json.load(f)

def savefig(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure -> {path}")

# ---- Fig 8: BSE Noise Robustness (Raw vs Denoised delay under noise) ----
p3 = load("experiments/phase3_belief_encoder/phase3_raw_results.json")
raw_delays  = [np.mean(p3[n]["raw_eval"]["avg_delay"])  for n in NOISE_SCENARIOS]
bse_delays  = [np.mean(p3[n]["bse_eval"]["avg_delay"])  for n in NOISE_SCENARIOS]
raw_stds    = [np.std(p3[n]["raw_eval"]["avg_delay"])   for n in NOISE_SCENARIOS]
bse_stds    = [np.std(p3[n]["bse_eval"]["avg_delay"])   for n in NOISE_SCENARIOS]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(NOISE_LEVELS, raw_delays, "o-", color="#FF9800", lw=2.2, ms=7, label="MAPPO-LCE (Raw Noisy Obs)")
ax.fill_between(NOISE_LEVELS, [d-s for d,s in zip(raw_delays,raw_stds)],
                [d+s for d,s in zip(raw_delays,raw_stds)], color="#FF9800", alpha=0.15)
ax.plot(NOISE_LEVELS, bse_delays, "s--", color="#00BCD4", lw=2.2, ms=7, label="MAPPO-LCE (BSE-Denoised)")
ax.fill_between(NOISE_LEVELS, [d-s for d,s in zip(bse_delays,bse_stds)],
                [d+s for d,s in zip(bse_delays,bse_stds)], color="#00BCD4", alpha=0.15)
ax.set_xticks(NOISE_LEVELS); ax.set_xticklabels(NOISE_SCENARIOS)
ax.set_xlabel("Noise Degradation Level (N5 Protocol)")
ax.set_ylabel("Average Vehicle Delay (s)")
ax.set_title("Figure 8: BSE Robustness — Raw Noisy Obs vs. BSE-Denoised Beliefs\n(MAPPO-LCE worker, same policy evaluated under two observation regimes)")
ax.legend()
savefig("experiments/phase3_belief_encoder/fig8_bse_noise_robustness.png")

# ---- Fig 9: BSE Calibration Detail (MSE + sigma-correlation) ----
calib = load("experiments/phase3_belief_encoder/bse_calibration_metrics.json")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Left: MSE comparison
bars = axes[0].bar(["Raw Obs\nMSE", "BSE Denoised\nMSE"],
                   [calib["raw_mse"], calib["bse_mse"]],
                   color=["#F44336", "#4CAF50"], width=0.45, edgecolor="white", alpha=0.9)
for bar, val in zip(bars, [calib["raw_mse"], calib["bse_mse"]]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
axes[0].set_ylabel("Mean Squared Error")
axes[0].set_title(f"BSE Denoising Quality\n({calib['mse_reduction_pct']:.1f}% MSE reduction)")

# Right: Sigma calibration as text summary
summary_text = (
    f"sigma-calibration correlation:\n"
    f"  r = {calib['sigma_calibration_correlation']:.3f}\n\n"
    f"Interpretation:\n"
    f"  sigma_hat positively correlates\n"
    f"  with actual estimation error.\n"
    f"  Larger sigma -> less confident\n"
    f"  estimate -> larger Lagrangian\n"
    f"  penalty (safety-aware).\n\n"
    f"Mean sigma_hat: {calib.get('mean_sigma', 0.5):.3f}"
)
axes[1].text(0.05, 0.95, summary_text, transform=axes[1].transAxes,
             fontsize=11, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8))
axes[1].axis("off")
axes[1].set_title("Uncertainty Calibration (Novelty N3)")

fig.suptitle("Figure 9: GRU-KalmanNet (BSE) — Denoising + Calibration Quality", fontsize=13)
savefig("experiments/phase3_belief_encoder/fig9_bse_calibration_detail.png")

# ---- Fig 5 (updated): Phase 5 Ablation ----
p5 = load("experiments/phase5_ablations/phase5_ablation_results.json")
abl_methods = list(p5.keys())
abl_delays  = [p5[m]["mean_delay"] for m in abl_methods]
abl_stds    = [np.std(p5[m]["avg_delay"]) for m in abl_methods]
COLOR_MAP = {
    "BSE (Proposed)": "#00BCD4", "BSE (No-Mod)": "#80DEEA",
    "Raw Obs": "#FF9800", "MAPPO": "#4CAF50",
    "FMA2C": "#9C27B0", "IPPO": "#2196F3", "Fixed-Time": "#555555"
}
def get_color(name):
    for k, v in COLOR_MAP.items():
        if k in name: return v
    return "#607D8B"

fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(abl_methods))
short_labels = [m.split(":")[1].strip() if ":" in m else m for m in abl_methods]
bars = ax.bar(x, abl_delays, yerr=abl_stds, capsize=4,
              color=[get_color(m) for m in abl_methods], edgecolor="white", alpha=0.9)
for bar, val in zip(bars, abl_delays):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.4,
            f"{val:.1f}s", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(short_labels, rotation=20, ha="right")
ax.set_ylabel("Average Vehicle Delay (s)")
ax.set_title("Figure 5: Phase 5 — Controlled Ablation Suite (Clean Condition)\n"
             "A0=Proposed, A1=No sigma-mod, A2=No BSE, A3=No constraints, A4=Hierarchical only, A5=Independent, A6=Fixed")
ax.axhline(abl_delays[-1], color="#555555", linestyle="--", lw=1.2, label="Fixed-Time baseline")
ax.legend()
savefig("experiments/phase5_ablations/fig5_ablation_comparison.png")

# ---- Fig 10: Phase 4 Full System Comparison ----
p4 = load("experiments/phase4_full_hierarchy/phase4_raw_results.json")
p4_methods = list(p4["clean"].keys())
P4_COLORS = {
    "Fixed-Time":                "#555555",
    "MAPPO (Flat)":              "#4CAF50",
    "FMA2C (Hier, No BSE)":      "#9C27B0",
    "MAPPO-LCE (Flat, No BSE)":  "#FF9800",
    "MAPPO-LCE + BSE (Proposed)":"#00BCD4",
}
fig, ax = plt.subplots(figsize=(9, 5))
for m in p4_methods:
    delays = [np.mean(p4[n][m]["avg_delay"]) for n in NOISE_SCENARIOS]
    stds   = [np.std(p4[n][m]["avg_delay"])  for n in NOISE_SCENARIOS]
    color  = P4_COLORS.get(m, "#607D8B")
    ls     = "--" if "Fixed-Time" in m else ("-." if "No BSE" in m and "FMA2C" in m else "-")
    lw     = 2.5 if "Proposed" in m else 2.0
    ax.plot(NOISE_LEVELS, delays, color=color, linestyle=ls, marker="o", lw=lw, ms=6, label=m)
    ax.fill_between(NOISE_LEVELS, [d-s for d,s in zip(delays,stds)],
                    [d+s for d,s in zip(delays,stds)], color=color, alpha=0.08)
ax.set_xticks(NOISE_LEVELS); ax.set_xticklabels(NOISE_SCENARIOS)
ax.set_xlabel("Noise Level (N5 Protocol)")
ax.set_ylabel("Average Vehicle Delay (s)")
ax.set_title("Figure 10: Phase 4 — Full System Comparison\n(Flat vs. Hierarchical vs. BSE-Enhanced)")
ax.legend(loc="upper left")
savefig("experiments/phase4_full_hierarchy/fig10_full_system_noise.png")

print("\nAll Phase 3-5 figures generated.")
