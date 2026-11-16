"""
Verify artifact integrity and compute SHA-256 hashes.
"""
import hashlib
import os

files_to_hash = [
    # V1 files
    "experiments/phase2_flat_baselines/phase2_raw_results.json",
    "experiments/phase2_flat_baselines/phase2_statistical_summary.json",
    "experiments/phase2_flat_baselines/robustness_slopes.json",
    # V2 files
    "experiments/phase2_canonical_v2/phase2_raw_results.json",
    "experiments/phase2_canonical_v2/phase2_statistical_summary.json",
    "experiments/phase2_canonical_v2/robustness_slopes.json",
    # Models
    "experiments/phase2_canonical_v2/models/ippo.pt",
    "experiments/phase2_canonical_v2/models/ippo_s43.pt",
    "experiments/phase2_canonical_v2/models/ippo_s44.pt",
    "experiments/phase2_canonical_v2/models/mappo.pt",
    "experiments/phase2_canonical_v2/models/mappo_s43.pt",
    "experiments/phase2_canonical_v2/models/mappo_s44.pt",
    "experiments/phase2_canonical_v2/models/mappo_lce.pt",
    "experiments/phase2_canonical_v2/models/mappo_lce_s43.pt",
    "experiments/phase2_canonical_v2/models/mappo_lce_s44.pt",
    "experiments/phase2_canonical_v2/models/fma2c.pt",
    "experiments/phase2_canonical_v2/models/fma2c_s43.pt",
    "experiments/phase2_canonical_v2/models/fma2c_s44.pt",
]

print("ARTIFACT SHA-256 HASH VERIFICATION:")
hashes = {}
for rel_path in files_to_hash:
    full_path = os.path.abspath(rel_path)
    if os.path.exists(full_path):
        with open(full_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        hashes[rel_path] = h
        print(f"{rel_path}: {h}")
    else:
        print(f"[MISSING] {rel_path}")


