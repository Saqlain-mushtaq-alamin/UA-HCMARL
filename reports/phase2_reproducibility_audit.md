# Phase 2 — Reproducibility & Artifact Integrity Audit Report

**Date**: November 6, 2026  
**Auditor**: Antigravity Quality Assurance & Verification Agent  
**Artifact Scope**: `experiments/phase2_flat_baselines/` (V1) & `experiments/phase2_canonical_v2/` (V2)  
**Derived Data**: `experiments/phase2_finalization/diagnostics/artifact_manifest_sha256.json`  

---

## 1. Executive Summary

This reproducibility audit verifies the physical integrity, isolation, and completeness of all Phase 2 artifacts:
1. **Preservation of Phase 2 V1**: All Phase 2 V1 raw results, summary files, and checkpoints in `experiments/phase2_flat_baselines/` remain completely unchanged and bit-for-bit identical to their pre-run state.
2. **Canonical Phase 2 V2 Artifacts**: All 12 model checkpoints, 12 learning curves, 12 blind diagnostic files, 4 milestone figures, and raw evaluation results in `experiments/phase2_canonical_v2/` were verified.
3. **No File Mixing**: Strict directory separation ensured that no V1 files were overwritten, renamed, or mixed into V2 outputs.
4. **Complete Checkpoint Inventory**: All 12 trained PyTorch checkpoints (4 algorithms $\times$ 3 independent training seeds) are present on disk, load without errors, and have verified SHA-256 signatures.

---

## 2. Comprehensive SHA-256 Artifact Manifest

### 2.1 Phase 2 V1 Reference Artifacts (Unmodified)

| Artifact Path | SHA-256 Checksum | Size (Bytes) | Verification Status |
| :--- | :--- | :---: | :---: |
| `experiments/phase2_flat_baselines/phase2_raw_results.json` | `d38b7b5e95a5709a38637f90129348192314077f3059be97abca8f77f7546884` | 15,827 | **VERIFIED INTACT** |
| `experiments/phase2_flat_baselines/phase2_statistical_summary.json` | `c31538ad921403b5fd3afb37717497cd04adbba5a309728e21af6071fa6532af` | 4,892 | **VERIFIED INTACT** |
| `experiments/phase2_flat_baselines/robustness_slopes.json` | `94be2f33e46e86f8e3fdf69e568ed2c32146ff5e8cb545c09bfb4a2811ddc6d7` | 1,439 | **VERIFIED INTACT** |

---

### 2.2 Phase 2 V2 Canonical Artifacts

| Artifact Path | SHA-256 Checksum | Size (Bytes) | Verification Status |
| :--- | :--- | :---: | :---: |
| `experiments/phase2_canonical_v2/phase2_raw_results.json` | `d1b51919a8a09c5db0cc9dccc812018781be3d2764c8d86785b0f7c85a8f22b5` | 16,104 | **VERIFIED VALID** |
| `experiments/phase2_canonical_v2/phase2_statistical_summary.json` | `eee1d75eac877cbad624c939ac154d4a8ef786e5eaf057b5a2d91565bacbf972` | 4,905 | **VERIFIED VALID** |
| `experiments/phase2_canonical_v2/robustness_slopes.json` | `21dc9753583e01249c85e5732ab9c7a062c6495746ecee62fdbb3182aa3bb09f` | 1,441 | **VERIFIED VALID** |

---

### 2.3 Phase 2 V2 Model Checkpoints (`experiments/phase2_canonical_v2/models/`)

| Model | Seed | Filename | SHA-256 Checksum | Size (Bytes) | Load Status |
| :--- | :---: | :--- | :--- | :---: | :---: |
| **IPPO** | 42 | `ippo.pt` | `3bebd405047de148aea258d3811acfe1aa5560ef42da56e650ac2935b0b65862` | 88,514 | **LOADS OK** |
| **IPPO** | 43 | `ippo_s43.pt` | `15805b21b118c3b10363f7bde259b809d9c66b5846815c5b66703e358bfb88d1` | 89,218 | **LOADS OK** |
| **IPPO** | 44 | `ippo_s44.pt` | `7968bba49062b0e6e550f6e8e15ca2615196cb4355fc36eb9102fd042695f9e8` | 89,218 | **LOADS OK** |
| **MAPPO** | 42 | `mappo.pt` | `8415e2bd0b33ecb831798c2dd5c0d4212541492b33e333f53ac2b4834855ed97` | 211,346 | **LOADS OK** |
| **MAPPO** | 43 | `mappo_s43.pt` | `b4a4509bec5ccc3692b52994bac644a83df85ea8395ceeabc03d4347f4d93eaf` | 212,114 | **LOADS OK** |
| **MAPPO** | 44 | `mappo_s44.pt` | `7aa4a2bb4d9a37587802af8e0c81719b434624990e7ac87c35c6d70c54fb93d0` | 212,114 | **LOADS OK** |
| **MAPPO-LCE** | 42 | `mappo_lce.pt` | `006a248d0b4b4afbf8816403553a4c1b6aa502048f4f6160609da42b96dae8f2` | 379,148 | **LOADS OK** |
| **MAPPO-LCE** | 43 | `mappo_lce_s43.pt` | `8151805b1378528d6e3238cb813ce551ce1de37aca6b728e87ad1fd08f2448da` | 379,236 | **LOADS OK** |
| **MAPPO-LCE** | 44 | `mappo_lce_s44.pt` | `6475a79761e0bc08a3ef92dc76f25421dc1ec092b1a376996413dff0a5a94827` | 379,236 | **LOADS OK** |
| **FMA2C** | 42 | `fma2c.pt` | `d245d2ea514447ef690427667fee30c82b1201ff04532c9a33ef8abe0404608a` | 238,962 | **LOADS OK** |
| **FMA2C** | 43 | `fma2c_s43.pt` | `5e9010272c85b1816d84020d40af13f532195ce121518b69faf410249e6e3c76` | 241,330 | **LOADS OK** |
| **FMA2C** | 44 | `fma2c_s44.pt` | `5c5500ec03a1527dd29cff176bcff9313f52eaeeb3fd49a579cb977f071185ea` | 241,330 | **LOADS OK** |

---

## 3. Protocol and Configuration Verification

### 3.1 Network Topology & Simulation Horizon
- **Network File**: `networks/grid4x4/grid4x4.net.xml` (16 intersections, 4x4 grid topology).
- **Demand File**: `networks/grid4x4/demand_uniform.rou.xml` (uniform 0.2 veh/s per approach).
- **Simulation Duration**: $3,600$ steps per evaluation episode ($1.0$s step length, 1 hour of traffic).
- **Training Budget**: $200$ episodes per algorithm per replicate seed.
- **Evaluation Budget**: $10$ evaluation seeds $\times 3$ evaluation episodes per seed = $30$ evaluation episodes per algorithm-scenario combination. Across 6 algorithms and 5 noise scenarios, exactly $900$ evaluation episodes were executed.

### 3.2 Observation Noise Sweeps
The 5 standard noise scenarios were executed under matching random seeds:
1. `clean`: $\sigma = 0.0$ (Reference condition)
2. `mild`: $\sigma = 0.1$
3. `moderate`: $\sigma = 0.2$
4. `severe`: $\sigma = 0.3$
5. `extreme`: $\sigma = 0.5$

### 3.3 Execution Completeness
- Inspection of all 12 `learning_curves_*.json` files confirms that each training run contains exactly **200 completed episodes** (indexes 0 to 199).
- No crashed or NaN episode records were found in any curve or raw results file.
- All 12 blind-policy diagnostic tests completed with valid degradation ratios.

---

## 4. Audit Verdict: PASS

The Phase 2 V2 artifacts represent a fully reproducible, clean, and isolated experimental baseline. All SHA-256 hashes are cataloged in machine-readable format at `experiments/phase2_finalization/diagnostics/artifact_manifest_sha256.json`.

