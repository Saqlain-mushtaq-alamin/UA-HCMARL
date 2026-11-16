# Phase 3 Independent Pre-Launch Verification Audit

**Auditor Role**: Independent Senior ML Systems & Experimental Methodology Auditor  
**Date**: November 6, 2026  
**Subject**: MARLTSC Phase 3 Pre-Launch Readiness  
**Target Benchmark**: Six-Arm Canonical Robustness Evaluation (Belief State Estimation)  
**Final Decision**: **NO-GO (CRITICAL METHODOLOGICAL & ARCHITECTURAL DEFECTS DETECTED)**  

---

## 1. Executive Summary & Audit Overview

As an independent auditor, I inspected the code, test suite, configurations, artifacts, and execution logs of Phase 3 in `D:\canvas\MARLTSC`. The implementation agent claimed that all four original blockers were resolved and that 63 tests pass.

### Verdict: **NO-GO**
While significant implementation progress has been achieved—namely the creation of a batched `BSEControllerWrapper`, a multi-noise training curriculum, and a disjoint seed allocation policy—**the Phase 3 benchmark cannot be certified for canonical execution**.

Independent inspection revealed four critical defects that invalidate the planned canonical experiment:
1. **Statistical Pseudo-Replication**: `scripts/run_phase3_benchmark.py` instantiates and evaluates only a single training seed (`seed=42`). The statistical summary computes Student's $t$-tests and Wilcoxon signed-rank tests across 10 evaluation traffic episodes from this single checkpoint, reviving the pseudo-replication defect that Phase 2 had to correct. Independent training replicates ($K=3$: seeds 42, 43, 44) are completely absent from the benchmark execution loop.
2. **BSE Kalman Recurrent Disconnect**: In `src/baselines/bse_wrapper.py:143`, `self.bse(hist_batch_t)` is called without `prev_belief`. Consequently, the transition network $f(\cdot)$ operates on the current noisy observation $o_t$ rather than the prior belief $\hat{\mathbf{b}}_{t-1}$, violating the published Kalman filter recurrence $\hat{\mathbf{b}}_t = (1 - \mathbf{K}_t) f(\hat{\mathbf{b}}_{t-1}) + \mathbf{K}_t \mathbf{o}_t$.
3. **Cross-Arm Object Mutation Risk**: In `scripts/run_phase3_benchmark.py:192 & 217`, `m3_ippo_bse` and `m6_fma2c_bse` wrap the exact same in-memory Python controller objects as `m1_ippo_clean` and `m4_fma2c_clean`. In `FMA2CController`, `self.current_goals` is mutated during action selection, causing cross-arm state pollution.
4. **Manifest Cryptographic Deficiency**: `phase3_manifest.json` does not compute SHA-256 hashes of generated checkpoints or raw result files, nor does it log git commit revision hashes.

---

## 2. Detailed Audit Findings by Task

### Task 1: BSE Controller Integration (`src/baselines/bse_wrapper.py`)

| Check Item | Verdict | Evidence & File Location | Analysis |
| :--- | :---: | :--- | :--- |
| **Observation & Decision Impact** | **PASS** | `bse_wrapper.py:148-165` | Empirically verified: action selection under noisy observations changed in 8/16 intersections when filtered through BSE. |
| **16-Intersection Processing** | **PASS** | `bse_wrapper.py:126-138` | Padded histories across all 16 agents are packed into shape `(16, 5, 16)` and processed in a single forward pass. |
| **Recurrent Hidden-State Reset** | **PASS** | `bse_wrapper.py:82-103` | `reset()` clears `self.histories`, `self.prev_beliefs`, and propagates reset to base controller. Verified via `test_bse_wrapper_reset_isolation`. |
| **Zero State Leakage** | **PASS** | `bse_wrapper.py:118-124` | Per-agent history lists prevent cross-intersection mixing; `reset()` prevents cross-episode leakage. |
| **Action Masking Effectiveness** | **PASS** | `bse_wrapper.py:164` | `action_masks` passed to base controller; verified in unit tests that illegal clearance actions are strictly zeroed out. |
| **Eval Mode & No Gradient** | **PASS** | `bse_wrapper.py:141-143` | `self.bse.eval()` and `with torch.no_grad():` strictly enforced. |
| **Checkpoint Compatibility** | **PASS** | `bse_wrapper.py:199-220` | Checkpoints save and load state dicts cleanly. |
| **Ground-Truth Isolation** | **PASS** | `bse_wrapper.py:105-115` | Operates strictly on `obs_dict`; no access to clean simulator attributes. |
| **Kalman Recurrence Pipeline** | **FAIL** | `bse_wrapper.py:143` | **CRITICAL DEFECT**: `self.bse(hist_batch_t)` omits `prev_belief`. `GRUKalmanNet.forward()` falls back to `prev_belief = current_obs`. Transition model is applied to noisy $o_t$ instead of $\hat{\mathbf{b}}_{t-1}$. |

---

### Task 2: Noise-Trained Baselines (`src/training/noise_training.py`)

| Check Item | Verdict | Evidence & File Location | Analysis |
| :--- | :---: | :--- | :--- |
| **Noise Injected During Training** | **PASS** | `noise_training.py:90` | `env.noise_injector.set_scenario(scenario)` called per episode. Verified non-zero perturbation ($> 0.64$ queue diff). |
| **All Presets Sampled** | **PASS** | `noise_training.py:86` | Uniform sampling across `["clean", "mild", "moderate", "severe", "extreme"]`. |
| **Noise Distribution Protocol** | **PASS** | `noise_training.py:24` | Uses canonical preset definitions matching Phase 2. |
| **RNG Independence** | **PASS** | `noise_training.py:71` | `noise_rng = np.random.default_rng(seed * 10007 + 777)` is decoupled from environment arrival RNG (`seed * 31 + i + 7919`). |
| **Policy Mapping** | **PASS** | `run_phase3_benchmark.py:173, 203` | Arm M2 maps to `IPPOController`; Arm M5 maps to `FMA2CController`. |
| **Training Budget Parity** | **PASS** | `run_phase3_benchmark.py:183, 213` | Canonical profile specifies 200 episodes $\times$ 3,600 steps, matching Phase 2. |
| **Checkpoint Storage Separation** | **PASS** | `run_phase3_benchmark.py:174, 204` | Models saved to `output_dir/models/ippo_noise.pt`, leaving Phase 2 artifacts untouched. |

---

### Task 3: Seed Allocation & Data Independence (`experiment_config.py`)

| Check Item | Verdict | Evidence & File Location | Analysis |
| :--- | :---: | :--- | :--- |
| **Train/Validation/Eval Disjointness** | **PASS** | `experiment_config.py:51-61` | BSE Train (`80042`), BSE Val (`81042`), Eval (`[1000, 1902]`), Train Replicates (`42, 43, 44`). Pairwise disjoint. |
| **Old Seed-1042 Elimination** | **PASS** | `experiment_config.py:25-26` | Seed 1042 completely removed from all configurations and collection routines. |
| **Formula-to-Runner Match** | **PASS** | `run_phase3_benchmark.py:243` | `ep_seed = base_seed + k` matches `cfg.get_eval_episode_seeds()`. |
| **Physical Trajectory Disjointness** | **PASS** | `test_seed_allocation.py:23-45` | Verified at simulator queue level: queue trajectories for 80042, 81042, and 1000 are distinct. |
| **Stream Logging in Manifest** | **PASS** | `phase3_manifest.json` | Manifest records `bse_train_seed`, `bse_test_seed`, and `eval_base_seeds`. |

---

### Task 4: Six-Arm Benchmark Runner (`scripts/run_phase3_benchmark.py`)

| Check Item | Verdict | Evidence & File Location | Analysis |
| :--- | :---: | :--- | :--- |
| **Exact Six-Arm Mapping** | **PASS** | `run_phase3_benchmark.py:220-226` | M1–M6 instantiated without legacy MAPPO-LCE baseline contamination. |
| **Matched Traffic Conditions** | **PASS** | `run_phase3_benchmark.py:249-268` | All 6 arms evaluated within the exact same episode loop on identical `ep_seed`. |
| **All 5 Noise Scenarios** | **PASS** | `run_phase3_benchmark.py:238` | Evaluates Clean, Mild, Moderate, Severe, Extreme sequentially. |
| **16 Intersections & 3600 Steps** | **PASS** | `experiment_config.py:88-91` | Config profile `'canonical'` enforces 16 intersections and 3600 steps. |
| **Atomic Checkpointing** | **PASS** | `checkpointed_runner.py:115-128` | Results written via temporary file + atomic rename. |
| **Training Replicate Support ($K=3$)** | **FAIL** | `run_phase3_benchmark.py:170-218` | **CRITICAL DEFECT**: Runner only instantiates seed 42. Seeds 43 and 44 are not evaluated. Statistical report runs on evaluation traffic seeds of a single model. |
| **Object Isolation Across Arms** | **FAIL** | `run_phase3_benchmark.py:192, 217` | **CRITICAL DEFECT**: `m3_ippo_bse` and `m6_fma2c_bse` share underlying controller instances with M1 and M4 in memory. Mutating `current_goals` in FMA2C creates state contamination. |
| **Manifest Checksums** | **FAIL** | `run_phase3_benchmark.py:365-385` | **CRITICAL DEFECT**: Manifest records output file basenames but lacks cryptographic SHA-256 hashes and git commit revision. |

---

### Task 5: Smoke-Test Evidence Verification (`experiments/phase3_smoke_test/`)

1. **Artifact Existence**:
   - `phase3_raw_results.json`: Present (10.7 KB), 5 scenarios $\times$ 6 arms $\times$ 2 seeds.
   - `bse_calibration_metrics.json`: Present ($0.1439 \to 0.1234$, $14.2\%$ reduction).
   - `phase3_statistical_summary.json`: Present.
   - `models/bse.pt` (99.7 KB), `models/ippo_noise.pt` (89.3 KB), `models/fma2c_noise.pt` (192.3 KB): Present.
   - 4 Figures: `fig1`–`fig4` generated cleanly.
2. **Inspection of Raw Results**:
   - In `experiments/phase3_smoke_test/phase3_raw_results.json`:
     - Under `clean`, M1, M2, and M3 had identical delays ($12.316$ and $9.199$).
     - In M6, across all 5 noise levels, delays were identical ($6.663$ and $4.359$).
   - **Audit Finding**: Under smoke settings (2 training episodes, 50 steps), undertrained IPPO defaults to holding phase 0. In M6, BSE denoised inputs so FMA2C's argmax policy took identical actions across all noise levels. Because SUMO is deterministic given a seed, identical actions produced identical delays. This explains the artifact behavior, but highlights the risk of relying on single-seed evaluations.
3. **Phase 2 Invariance**:
   - Verified that `experiments/phase2_canonical_v2/` files were untouched.

---

### Task 6: Scientific Protocol and Statistical Methodology

1. **Unit of Statistical Independence**:
   - The primary methodological failure in the current runner is treating evaluation episodes from a single trained model as independent experimental units.
   - If only seed 42 is evaluated, the sample size for hypothesis testing is $N=10$ evaluation episodes, NOT $K=3$ independent model training replicates.
   - This reintroduces pseudo-replication: variations across evaluation episodes reflect traffic noise, not algorithmic variance across training runs.
2. **Overstated Claims**:
   - The implementation report claimed all blockers were resolved and the system was ready for canonical benchmark. This claim was premature given the $K=1$ training replicate limit and the missing Kalman recurrence update in `bse_wrapper.py`.

---

## 3. Comprehensive Finding Summary

| Finding ID | Task | Component | Severity | Verdict | Description |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **F-01** | Task 4 / 6 | `run_phase3_benchmark.py` | **BLOCKER** | **FAIL** | Runner only executes seed 42 ($K=1$); training replicate seeds 43 and 44 are omitted. Statistical tests evaluate traffic seeds, committing pseudo-replication. |
| **F-02** | Task 1 | `bse_wrapper.py:143` | **BLOCKER** | **FAIL** | `prev_belief` is omitted in `bse.forward()`. Transition model predicts from current noisy observation $o_t$ instead of prior belief $\hat{\mathbf{b}}_{t-1}$. |
| **F-03** | Task 4 | `run_phase3_benchmark.py:192, 217` | **BLOCKER** | **FAIL** | M3 and M6 share in-memory controller instances with M1 and M4. FMA2C mutates `current_goals` in-place, causing cross-arm contamination. |
| **F-04** | Task 7 | `run_phase3_benchmark.py:365` | **BLOCKER** | **FAIL** | Experiment manifest lacks SHA-256 cryptographic hashes for artifacts/checkpoints and git commit SHA. |
| **F-05** | Task 2 | `noise_training.py` | MINOR | **PASS** | Noise injection during training is functional, reproducible, and decoupled from traffic seeds. |
| **F-06** | Task 3 | `experiment_config.py` | MINOR | **PASS** | Centralized seed allocation is mathematically and trajectory-level disjoint. |
| **F-07** | Task 1 | `bse_wrapper.py` | MINOR | **PASS** | Action masking and batched tensor execution across 16 intersections verified functional. |

---

## 4. Final Audit Decision

### Launch Authorization Decision: **NO-GO**

The implementation agent resolved the superficial execution issues (the smoke test passes and 63 tests pass), but introduced or left unresolved four critical architectural, methodological, and scientific defects.

Phase 3 is **NOT READY** for canonical execution. Launching now would generate invalid scientific conclusions based on pseudo-replicated statistics and an un-recurrent belief filter.

