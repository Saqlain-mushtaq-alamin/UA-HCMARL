"""
Inspect all raw results from canonical Phase 2 V2 run.
"""
import json
import numpy as np

with open("experiments/phase2_canonical_v2/phase2_raw_results.json", "r") as f:
    raw = json.load(f)

scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
methods = ["Fixed-Time", "Actuated", "IPPO", "MAPPO", "MAPPO-LCE", "FMA2C"]

print("==================================================================")
print("CANONICAL PHASE 2 V2 RAW RESULTS AUDIT")
print("==================================================================")

for m in methods:
    print(f"\nMethod: {m}")
    for s in scenarios:
        delays = raw[s][m]["avg_delay"]
        viols = raw[s][m]["violation_rate"]
        mean_d = np.mean(delays)
        std_d = np.std(delays)
        mean_v = np.mean(viols)
        print(f"  [{s:8s}] Queue: {mean_d:6.2f} +/- {std_d:5.2f} (n={len(delays)}) | Viols: {mean_v*100:6.3f}% | Raw queues: {[round(x, 2) for x in delays]}")


