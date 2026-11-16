import json

with open('experiments/phase3_canonical/phase3_statistical_summary.json') as f:
    d = json.load(f)

for sc in ['clean', 'mild', 'moderate', 'severe', 'extreme']:
    print(f'=== {sc.upper()} ===')
    for m in [
        'M1: IPPO (Clean)',
        'M2: IPPO (Noise-Trained)',
        'M3: IPPO + BSE (Ours)',
        'M4: FMA2C (Clean)',
        'M5: FMA2C (Noise-Trained)',
        'M6: FMA2C + BSE (Ours)'
    ]:
        q = d['scenarios'][sc][m]['queue_length']
        reps = [round(x, 2) for x in q['replicate_means']]
        print(f"{m:26s}: Mean = {q['mean']:6.2f} +/- {q['std']:5.2f} | Replicates: {reps}")
    print()
