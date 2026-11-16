"""
Detailed breakdown of MAPPO per training seed and scenario.
"""
import json
import numpy as np

with open("experiments/phase2_canonical_v2/phase2_raw_results.json", "r") as f:
    raw = json.load(f)

scenarios = ["clean", "mild", "moderate", "severe", "extreme"]

print("==========================================================")
print("MAPPO PER-TRAINING SEED BREAKDOWN")
print("==========================================================")

# In run_phase2_benchmark.py:
# eval seed 0 -> model s42
# eval seed 1 -> model s43
# eval seed 2 -> model s44
# eval seed 3 -> model s42
# eval seed 4 -> model s43
# eval seed 5 -> model s44
# eval seed 6 -> model s42
# eval seed 7 -> model s43
# eval seed 8 -> model s44
# eval seed 9 -> model s42

for scen in scenarios:
    delays = raw[scen]["MAPPO"]["avg_delay"]
    viols = raw[scen]["MAPPO"]["violation_rate"]
    
    # Note: earlier we noticed tasks were submitted and recorded via as_completed.
    # Let's check whether delays in the list correspond to index i or if there's seed variation.
    print(f"\nScenario: {scen.upper()}")
    print(f"  All 10 delays: {[round(x, 3) for x in delays]}")
    print(f"  All 10 viols:  {[round(x*100, 4) for x in viols]}")
    
    # Grouping by modulo 3
    s42 = [delays[i] for i in range(len(delays)) if i % 3 == 0]
    s43 = [delays[i] for i in range(len(delays)) if i % 3 == 1]
    s44 = [delays[i] for i in range(len(delays)) if i % 3 == 2]
    
    print(f"  Mod 0 (s42) mean: {np.mean(s42):.3f} +/- {np.std(s42):.3f} (n={len(s42)})")
    print(f"  Mod 1 (s43) mean: {np.mean(s43):.3f} +/- {np.std(s43):.3f} (n={len(s43)})")
    print(f"  Mod 2 (s44) mean: {np.mean(s44):.3f} +/- {np.std(s44):.3f} (n={len(s44)})")

