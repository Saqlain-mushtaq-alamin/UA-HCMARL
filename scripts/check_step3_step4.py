"""
Step 3: Identical-array check on new phase5_raw_results.json
Step 4: Sanity-check Cohen's d for implausible effect sizes (|d| > 5)
"""
import json
import numpy as np

print("=" * 60)
print("  Step 3: Identical-Array Check")
print("=" * 60)

with open('experiments/phase5_ablations/phase5_raw_results.json') as f:
    d = json.load(f)

variants = list(d['clean'].keys())
print(f"Variants found: {variants}\n")

any_identical = False
for nl in ['clean', 'mild', 'moderate', 'severe', 'extreme']:
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a = d[nl][variants[i]]['avg_delay']
            b = d[nl][variants[j]]['avg_delay']
            if a == b:
                # A0 vs A3 under clean is expected (documented architectural inertia)
                is_expected = (
                    nl == 'clean' and
                    'A0' in variants[i] and 'A3' in variants[j]
                ) or (
                    nl == 'clean' and
                    'A3' in variants[i] and 'A0' in variants[j]
                )
                tag = "(expected -- documented A0~A3 clean-condition inertia)" if is_expected else "<-- BUG: STILL IDENTICAL"
                print(f"  [{nl}] {variants[i]} == {variants[j]}  {tag}")
                if not is_expected:
                    any_identical = True

if not any_identical:
    print("  [OK] No unexpected identical arrays. All variant pairs differ.\n")
else:
    print("\n  [FAIL] Unexpected identical arrays remain -- investigate before trusting results.\n")

print("=" * 60)
print("  Step 4: Cohen's d Implausibility Check (|d| > 5 = suspicious)")
print("=" * 60)

with open('experiments/phase5_ablations/phase5_statistical_matrix.json') as f:
    stats = json.load(f)

any_suspicious = False
for k, v in stats.items():
    cd = v['cohen_d']
    flag = "  <-- SUSPICIOUS (near-zero variance)" if abs(cd) > 5 else ""
    print(f"  {k:40s}  d={cd:8.3f}  p={v['p_value']:.5f}{flag}")
    if abs(cd) > 5:
        any_suspicious = True

print()
if not any_suspicious:
    print("  [OK] No |Cohen's d| > 5 anomalies. Effect sizes are plausible.\n")
else:
    print("  [WARN] Some comparisons show implausibly large effect sizes.\n"
          "    This indicates near-zero-variance arrays -- investigate those rows.\n")

# Also show full descriptive stats for transparency
print("=" * 60)
print("  Descriptive statistics summary (severe noise, all variants)")
print("=" * 60)
for v_name in variants:
    delays = d['severe'][v_name]['avg_delay']
    print(f"  {v_name:40s}  mean={np.mean(delays):7.2f}  std={np.std(delays):6.2f}  n={len(delays)}")
