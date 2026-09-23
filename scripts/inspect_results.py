import json
import numpy as np

with open('experiments/phase5_ablations/phase5_raw_results.json') as f:
    d = json.load(f)

for nl in ['clean', 'mild', 'moderate', 'severe', 'extreme']:
    print(f"=== Noise: {nl} ===")
    for k, v in d[nl].items():
        delay = np.array(v['avg_delay'])
        viol = np.array(v['violation_rate'])
        print(f"  {k:30s} | Delay: {delay.mean():6.2f} +/- {delay.std():4.2f} | Viol: {viol.mean()*100:5.1f}%")
    print()
