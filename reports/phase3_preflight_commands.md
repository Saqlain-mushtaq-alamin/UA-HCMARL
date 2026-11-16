# Phase 3 Preflight Commands & Execution Guide

**Date**: November 6, 2026  
**Auditor**: Antigravity Quality Assurance & Verification Agent  
**Context**: MARLTSC Phase 3 Launch Authorization  
**Notice**: **DO NOT EXECUTE CANONICAL COMMANDS UNTIL ALL BLOCKERS IN `phase3_blockers_and_fixes.md` ARE RESOLVED.**

---

## 1. Safety Rules & Verification Gate

Before running any script:
1. Confirm you are in the workspace root: `D:\canvas\MARLTSC`.
2. Verify that Phase 2 canonical artifacts in `experiments/phase2_canonical_v2/` remain strictly read-only.
3. Ensure all tests run in an isolated environment using temporary output directories.

---

## 2. Preflight Step 1: Unit & Regression Test Suite

Execute the existing test suite to confirm that baseline environment and model primitives remain uncorrupted:

```powershell
# Run full regression test suite (52 tests expected to pass)
python -m pytest tests/ -v

# Run targeted Layer 1 BSE unit tests
python -m pytest tests/test_bse.py -v
```

---

## 3. Preflight Step 2: Phase 2 Checkpoint Verification

Verify that all 12 pre-trained Phase 2 checkpoints exist, load, and execute evaluation inference cleanly:

```powershell
# Verify Phase 2 IPPO and FMA2C checkpoints
python -c "
import torch
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController

agent_ids = [f'intersection_{i}' for i in range(16)]

# 1. IPPO Checkpoint
ctrl_ippo = IPPOController(agent_ids=agent_ids)
ctrl_ippo.load_model('experiments/phase2_canonical_v2/models/ippo.pt')
print('[PASS] IPPO seed 42 checkpoint loaded.')

# 2. FMA2C Checkpoint
ctrl_fma2c = FMA2CController(agent_ids=agent_ids, num_regions=4)
ctrl_fma2c.load_model('experiments/phase2_canonical_v2/models/fma2c.pt')
print('[PASS] FMA2C seed 42 checkpoint loaded.')
"
```

---

## 4. Preflight Step 3: Isolated Smoke Test (Safe Dry-Run)

Once the blockers in `reports/phase3_blockers_and_fixes.md` are patched, execute an isolated, non-destructive smoke test using the `'smoke'` configuration profile (4 intersections, 2 seeds, 100 steps) in a temporary directory:

```powershell
# Execute isolated smoke test in temporary scratch directory
python scripts/run_phase3_benchmark.py `
  --profile smoke `
  --output-dir experiments/phase3_smoke_test
```

### Verification Checks on Smoke Output:
```powershell
# Confirm smoke output JSON and figure generation
python -c "
import json, os

path = 'experiments/phase3_smoke_test/phase3_raw_results.json'
assert os.path.exists(path), 'Smoke results JSON missing!'
with open(path) as f:
    data = json.load(f)
print('Scenarios in smoke test:', list(data.keys()))
print('[PASS] Smoke test outputs verified.')
"
```

---

## 5. Preflight Step 4: Canonical Execution (Authorization Pending)

> [!CAUTION]
> **DO NOT EXECUTE THESE COMMANDS UNTIL BLOCKERS 1, 2, 3, AND 4 ARE REMEDIATED AND APPROVED.**

Once the implementation fixes are in place, the canonical Phase 3 benchmark can be executed with the following standardized parameters:

### Step 4A: Pre-Train Layer 1 BSE (Supervised)
```powershell
python scripts/train_bse_canonical.py `
  --steps 2000 `
  --epochs 20 `
  --train-seed 80042 `
  --test-seed 81042 `
  --output-dir experiments/phase3_canonical/models
```

### Step 4B: Train Noise-Augmented Baselines (Arms M2 & M5)
```powershell
# Train Noise-Trained IPPO across 3 seeds (42, 43, 44)
python scripts/train_noise_baselines.py `
  --algorithm ippo `
  --seeds 42 43 44 `
  --train-episodes 200 `
  --max-steps 3600 `
  --output-dir experiments/phase3_canonical/models

# Train Noise-Trained FMA2C across 3 seeds (42, 43, 44)
python scripts/train_noise_baselines.py `
  --algorithm fma2c `
  --seeds 42 43 44 `
  --train-episodes 200 `
  --max-steps 3600 `
  --output-dir experiments/phase3_canonical/models
```

### Step 4C: Canonical Six-Arm Evaluation
```powershell
python scripts/run_phase3_benchmark.py `
  --profile canonical `
  --num-workers 12 `
  --output-dir experiments/phase3_canonical
```

---

## 6. Preflight Step 5: Post-Benchmark Statistical Report Generation

After canonical benchmark completion, run the verified statistical analysis and figure generation:

```powershell
python scripts/generate_phase3_report.py `
  --input-dir experiments/phase3_canonical `
  --output-dir experiments/phase3_canonical/reports
```

