"""
Statistical Evaluation & Significance Testing Script (Phase 1 §1.7)

Analyzes raw baseline run outputs and outputs formatted statistical tables.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.statistical import generate_statistical_report


def analyze_results(results_file: str = "experiments/phase1_baselines/baseline_raw_results.json"):
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found. Run scripts/run_baselines.py first.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    print("\n========================================================")
    print("      Statistical Significance Analysis Report         ")
    print("========================================================")

    for demand, noise_dict in data.items():
        print(f"\n### Demand Profile: {demand.upper()}")
        for noise, methods in noise_dict.items():
            print(f"\n--- Scenario: {noise.capitalize()} ---")
            report = generate_statistical_report(methods, baseline_name="fixed_time")
            
            for m_name, stats_info in report.items():
                mean = stats_info["mean"]
                low = stats_info["ci_95_low"]
                high = stats_info["ci_95_high"]
                print(f"  * {m_name:12s}: Mean = {mean:6.2f} s | 95% CI = [{low:6.2f}, {high:6.2f}]")
                
                if "vs_baseline" in stats_info:
                    vs = stats_info["vs_baseline"]
                    p_val = vs["p_value"]
                    sig = "Yes (p < 0.05)" if vs["significant_p_lt_005"] else "No (p >= 0.05)"
                    cohen = vs["cohens_d"]
                    print(f"    -> vs Fixed-Time: Wilcoxon p-value = {p_val:.4f} | Significant: {sig} | Cohen's d = {cohen:.2f}")


if __name__ == "__main__":
    analyze_results()
