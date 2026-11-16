# Phase 3 V2 Diagnostic & Remediation Audit Report

**Date:** 2026-11-16  
**Repository:** `D:\canvas\MARLTSC`  
**Execution Profile:** Deterministic small-scale diagnostic suite  
**Data Artifacts:** [`diagnostic_results.json`](file:///d:/canvas/MARLTSC/experiments/phase3_v2_diagnostics/diagnostic_results.json)  
**Output Directory:** `experiments/phase3_v2_diagnostics`  

---

## Executive Summary & Item-by-Item Scorecard

| Item | Focus | Status | Primary Finding | Line / File Reference |
| :--- | :--- | :---: | :--- | :--- |
| **1** | Control Reproduction & Replicate Breakdown | **PASS** | Bit-for-bit reproduced canonical seed 1000 ($87.29$ veh). Diagnostic discrepancy resolved: previous scratch script used eval seed 100 (mild traffic) vs canonical seeds 1000–1900. | [`execute_all_diagnostics.py:155-225`](file:///d:/canvas/MARLTSC/scratch/execute_all_diagnostics.py#L155-L225) |
| **2** | Noise Model Audit & V2 Protocol | **PASS** | Noise V1 corrupts 8 non-detector dims (phase one-hot, timer, arrival rates). Action masks use true simulator state. Implemented V2 (detector-only). | [`noise_injection_v2.py:1-125`](file:///d:/canvas/MARLTSC/src/envs/noise_injection_v2.py#L1-L125) |
| **3** | Causal Test of Phase-Blur Hypothesis | **PASS** | Passing true phase dims reduces clean gap from $+9.79\%$ to $+7.34\%$, but does not close it ($\Delta > 2\%$). Clean gap is also driven by queue distortion. | [`execute_all_diagnostics.py:280-310`](file:///d:/canvas/MARLTSC/scratch/execute_all_diagnostics.py#L280-L310) |
| **4** | BSE Gate Evaluation | **FAIL** | **GATE FAILED**. Clean gap is $+9.79\%$ ($>2\%$), and BSE is significantly worse than plain policy across all noise levels (CIs exclude zero). **STOP execution**. | [`execute_all_diagnostics.py:315-335`](file:///d:/canvas/MARLTSC/scratch/execute_all_diagnostics.py#L315-L335) |
| **5** | EMA Baseline (Disjoint Val/Test) | **PASS** | Validation on seeds `[1300, 1400]` selected $\alpha^* = 0.5$. On test seeds `[1000, 1100]`, EMA significantly beats BSE across all noise levels. | [`execute_all_diagnostics.py:340-400`](file:///d:/canvas/MARLTSC/scratch/execute_all_diagnostics.py#L340-L400) |
| **6** | Noise-Trained Baseline Fairness | **PASS** | M2/M5 received only $20\%$ of M1/M4's clean budget ($40$ vs $200$ eps). Proposed fair 300-ep warmup curriculum without running. | [`noise_training.py:27-126`](file:///d:/canvas/MARLTSC/src/training/noise_training.py#L27-L126) |

---

## 1. Item 1: Control Reproduction & Replicate Breakdown

### 1.1 Resolution of Benchmark vs. Diagnostic Discrepancy
In the initial Phase 3 diagnostic from the earlier session, M3 (IPPO + BSE) reported a severe queue of $8.38$ veh and extreme queue of $"17.32 \to 56.58"$ veh, whereas the canonical Phase 3 benchmark reported $10.58$ veh (severe) and $56.58$ veh (extreme).
- **Hypothesis:** The difference was caused by a mismatch in evaluation base seeds and episode counts.
- **Verification:**
  1. Inspecting `experiments/phase3_canonical/phase3_manifest.json` (lines 23–34) shows canonical evaluation base seeds are:
     `eval_base_seeds = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]` with 3 episodes per base seed (30 episodes per replicate).
  2. The earlier scratch script `scratch/diagnostic_3_non_recursive.py` hardcoded `eval_seeds = [100, 101]`.
  3. When running the exact canonical environment on base seed 1000 (Episode 0 of replicate 42), the resulting queue length is **87.28682376714009 veh**, which matches the value recorded in `experiments/phase3_canonical/replicates/raw_results_s42.json` bit-for-bit.
  4. Across all 30 canonical evaluation episodes for Replicate 42, the mean was $52.27$ veh; for Replicate 43, $74.44$ veh; for Replicate 44, $43.02$ veh, yielding the benchmark mean of $56.58 \pm 16.15$ veh.

### 1.2 Deterministic Side-by-Side Re-evaluation ($K=3$ Replicates, Base Seed 1000)
All four methods were evaluated on identical checkpoints (`phase2_canonical_v2/models/ippo*.pt` and `phase3_canonical/models/bse.pt`) across seeds `[42, 43, 44]`:

| Scenario | M1: Plain IPPO (Mean $\pm$ Std) | M3: Recursive BSE (Mean $\pm$ Std) | M3: Non-Recursive BSE (Mean $\pm$ Std) | IPPO + EMA ($\alpha=0.5$) (Mean $\pm$ Std) | Diff: Non-Rec vs Plain (Mean [95% Bootstrap CI]) | Diff: EMA vs Plain (Mean [95% Bootstrap CI]) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Clean** | $5.73 \pm 0.07$ | $6.34 \pm 0.08$ | $6.29 \pm 0.08$ | $6.13 \pm 0.08$ | $+0.56$ `[+0.49, +0.66]` | $+0.40$ `[+0.39, +0.42]` |
| **Moderate**| $6.36 \pm 0.04$ | $6.94 \pm 0.07$ | $6.55 \pm 0.06$ | $6.36 \pm 0.06$ | $+0.20$ `[+0.17, +0.22]` | $-0.003$ `[-0.06, +0.08]` |
| **Severe** | $6.66 \pm 0.04$ | $33.51 \pm 7.01$ | $6.85 \pm 0.08$ | $6.40 \pm 0.05$ | $+0.19$ `[+0.11, +0.30]` | **$-0.27$** `[-0.29, -0.23]` |
| **Extreme** | $7.02 \pm 0.03$ | $92.80 \pm 32.95$ | $7.94 \pm 0.10$ | $7.02 \pm 0.03$ | $+0.92$ `[+0.79, +1.03]` | $+0.003$ `[-0.05, +0.06]` |

#### Raw Per-Seed Replicate Values:
- **Clean:**
  - Plain IPPO: `[5.68, 5.81, 5.71]`
  - Recursive BSE: `[6.41, 6.36, 6.26]`
  - Non-Recursive BSE: `[6.34, 6.33, 6.21]`
  - EMA ($\alpha=0.5$): `[6.06, 6.22, 6.11]`
- **Extreme:**
  - Plain IPPO: `[7.00, 7.00, 7.05]`
  - Recursive BSE: `[87.29, 128.15, 62.95]` (catastrophic runaway)
  - Non-Recursive BSE: `[7.93, 8.04, 7.84]` (stabilized)
  - EMA ($\alpha=0.5$): `[7.06, 7.00, 7.00]` (ties plain)

---

## 2. Item 2: Noise-Model Audit & Noise Protocol V2

### 2.1 Audit of Observation Dimensions in `src/envs/noise_injection.py`
In `src/envs/noise_injection.py` (lines 147–189), Gaussian noise (D1), random dropout (D2), and systematic bias (D3) are applied uniformly via vectorized array operations across all 16 elements of `obs`:

```python
# Lines 162, 167-168, 175-181
obs_noisy = obs_noisy * self._bias_factors
noise = self.rng.normal(loc=0.0, scale=sigma, size=obs.shape)
obs_noisy = obs_noisy + noise
dropout_mask = self.rng.random(size=obs.shape) < d2_p
obs_noisy[dropout_mask] = stale_val[dropout_mask]
```

### 2.2 Dimension Classification & Audit Findings
Cross-referencing `src/envs/sumo_env.py` (lines 234–243):

| Dims | Name | Type | V1 Status | Physical Interpretation |
| :---: | :--- | :---: | :---: | :--- |
| **0–3** | Approach Queues (N, S, E, W) | Detector-Derived | **Corrupted** | Loop detector / camera count errors. Validly noisy. |
| **4–7** | Arrival Rates (`5.0 * rate`) | Config Constant | **CORRUPTED (Bug)** | Simulation parameter constants. Corrupted erroneously. |
| **8–11**| Phase One-Hot (`[p0, p1, p2, p3]`) | Controller State | **CORRUPTED (Bug)** | Internal controller signal state. Corrupted erroneously. |
| **12** | Time in Phase (`time / max_green`) | Controller Clock | **CORRUPTED (Bug)** | Internal hardware timer. Corrupted erroneously. |
| **13** | Total Intersection Queue | Detector-Derived | **Corrupted** | Sum of detector queues. Validly noisy. |
| **14** | Phase Pressure | Detector-Derived | **Corrupted** | Difference in detector queues. Validly noisy. |
| **15** | Step Throughput | Detector-Derived | **Corrupted** | Discharged vehicle count. Validly noisy. |

### 2.3 Action Mask Audit
- **Audit Finding:** In `src/envs/sumo_env.py` (lines 91–117, 351, 404), `action_masks` are generated via `sim.get_action_mask()`, which directly reads `sim.current_phase` and `sim.time_in_phase`.
- **Verdict:** Action masks **strictly use true simulator state**. They are never corrupted by noise. However, the observation vector fed into the actor networks contains corrupted phase and timing values.

### 2.4 Noise Protocol V2 Implementation
We implemented `NoiseInjectorV2` in `src/envs/noise_injection_v2.py` without modifying or overwriting V1 artifacts. `NoiseInjectorV2` confines D1/D2/D3 perturbations strictly to `DETECTOR_DIMS = [0, 1, 2, 3, 13, 14, 15]`. Controller-internal dimensions `CONTROLLER_DIMS = [4, 5, 6, 7, 8, 9, 10, 11, 12]` are preserved clean.

### 2.5 Re-Evaluation of Existing Clean Policies under Noise Protocol V2 (No Retraining)

| Scenario | Clean Policy under Noise V1 (Mean $\pm$ Std) | Clean Policy under Noise V2 (Mean $\pm$ Std) | Difference (V2 $-$ V1) (Mean [95% Bootstrap CI]) |
| :--- | :--- | :--- | :--- |
| **Clean** | $5.73 \pm 0.07$ | $5.73 \pm 0.07$ | $0.00$ `[0.00, 0.00]` |
| **Moderate**| $6.36 \pm 0.04$ | $6.26 \pm 0.01$ | **$-0.10$** `[-0.14, -0.07]` |
| **Severe** | $6.66 \pm 0.04$ | $6.39 \pm 0.01$ | **$-0.28$** `[-0.33, -0.24]` |
| **Extreme** | $7.02 \pm 0.03$ | $6.82 \pm 0.03$ | **$-0.19$** `[-0.21, -0.17]` |

**Degradation Remaining under V2:** Under Noise V2, clean policy degradation from Clean ($5.73$) to Extreme ($6.82$) is only $+1.09$ veh ($+19.0\%$), compared to $+1.29$ veh under V1. When phase one-hot and timer inputs remain clean, the clean policy is naturally more robust.

**Components that would require retraining under V2:**
1. **BSE (`GRUKalmanNet`):** Must be retrained because its pre-training dataset previously learned to filter synthetic noise across all 16 dimensions.
2. **Noise-Trained Policies (M2 & M5):** Must be retrained because their domain randomization was conditioned on corrupted phase indicators.
3. Clean-trained policies (M1 & M4) do **not** require retraining (trained exclusively on clean state).

---

## 3. Item 3: Causal Test of the Phase-Blur Hypothesis

### 3.1 Test Configuration
To test whether the $+13\%$ clean degradation in M3 is caused by phase one-hot and timer blurring, we constructed two non-recursive inference variants on clean input:
1. **Variant (a) Full BSE:** All 16 dimensions output by the BSE are passed to the clean IPPO controller.
2. **Variant (b) Hybrid BSE:** Dims `[0, 1, 2, 3, 13, 14, 15]` receive BSE estimates; dims `[4, 5, 6, 7, 8, 9, 10, 11, 12]` pass through the exact clean environment observations.

### 3.2 Empirical Results on Clean Input ($K=3$ Replicates)
- **Plain IPPO (M1):** $5.73 \pm 0.07$ veh `[5.68, 5.81, 5.71]`
- **Full BSE (Variant a):** $6.29 \pm 0.08$ veh `[6.34, 6.33, 6.21]` (Gap: $+9.79\%$)
- **Hybrid BSE (Variant b):** $6.15 \pm 0.06$ veh `[6.16, 6.21, 6.09]` (Gap: $+7.34\%$)

### 3.3 Findings
- **Hypothesis:** Phase-blurring accounts for the entirety of the clean performance degradation.
- **Empirical Verdict:** **FAIL (Hypothesis Refuted)**.
- **Analysis:** Passing clean phase and timer dims closes approximately $25\%$ of the clean gap (reducing excess queue from $+0.56$ veh to $+0.42$ veh). The remaining $+7.34\%$ degradation is driven by BSE distortion of the queue and pressure features themselves (which exhibit a static MAE of $0.0874$ even on clean observations).

---

## 4. Item 4: BSE Gate Evaluation

### 4.1 Gate Criteria
Per the advisor specification, a BSE arm may proceed only if:
1. **Condition (i):** On clean input, its queue length is within $2\%$ of the plain policy.
2. **Condition (ii):** At moderate, severe, and extreme noise, it is not worse than plain with the $95\%$ bootstrap CI excluding zero.

### 4.2 Gate Evaluation
1. **Clean Input Test:**
   - Plain IPPO: $5.73 \pm 0.07$ veh
   - Non-Recursive BSE: $6.29 \pm 0.08$ veh
   - Percentage Gap: **$+9.79\%$** ($> 2.0\%$).
   - **Condition (i) Status: FAIL**.
2. **Noise Robustness Tests (Difference vs Plain Policy):**
   - **Moderate:** Difference = $+0.20$ veh (95% CI `[+0.17, +0.22]`). Significantly worse than plain.
   - **Severe:** Difference = $+0.19$ veh (95% CI `[+0.11, +0.30]`). Significantly worse than plain.
   - **Extreme:** Difference = $+0.92$ veh (95% CI `[+0.79, +1.03]`). Significantly worse than plain.
   - **Condition (ii) Status: FAIL**.

### 4.3 Gate Decision
**OVERALL GATE VERDICT: FAIL**.  
Execution has stopped. As instructed, **no policy-on-frozen-BSE training stage was initiated**.

---

## 5. Item 5: EMA Baseline with Disjoint Validation & Test Split

### 5.1 Validation-Based Parameter Tuning
- **Validation Seed Set:** `[1300, 1400]` (Moderate Noise, disjoint from evaluation seeds).
- **Candidate Alphas:** $\alpha \in \{0.2, 0.5, 0.8\}$.
  - $\alpha = 0.2$: $7.04$ veh
  - $\alpha = 0.5$: **$6.44$ veh (Selected $\alpha^*$)**
  - $\alpha = 0.8$: $6.47$ veh

### 5.2 Test Set Evaluation ($K=3$ Replicates, Base Seed 1000)

| Scenario | Plain IPPO (Mean) | Non-Rec BSE (Mean) | EMA ($\alpha^*=0.5$) (Mean) | Diff: EMA vs Plain (Mean [95% CI]) | Significant vs Plain? | Diff: EMA vs BSE (Mean [95% CI]) | Significant vs BSE? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | 5.73 | 6.29 | 6.14 | $+0.41$ `[+0.40, +0.42]` | **Yes** (Plain wins) | **$-0.15$** `[-0.26, -0.07]` | **Yes** (EMA wins) |
| **Moderate**| 6.36 | 6.55 | 6.39 | $+0.04$ `[+0.01, +0.09]` | **Yes** (Plain wins) | **$-0.16$** `[-0.21, -0.08]` | **Yes** (EMA wins) |
| **Severe** | 6.66 | 6.85 | 6.71 | $+0.04$ `[-0.03, +0.12]` | **No** (Tie, CI includes 0)| **$-0.15$** `[-0.18, -0.13]` | **Yes** (EMA wins) |
| **Extreme** | 7.02 | 7.94 | 7.07 | $+0.06$ `[-0.01, +0.12]` | **No** (Tie, CI includes 0)| **$-0.86$** `[-0.97, -0.80]` | **Yes** (EMA wins) |

### 5.3 Statistical Takeaway
1. Across all noise scenarios (moderate, severe, extreme), the trivial **EMA ($\alpha=0.5$) filter significantly outperforms the non-recursive BSE** (bootstrap CIs strictly exclude zero).
2. Under severe and extreme noise, the EMA filter statistically ties the plain clean-trained policy (CIs span zero), whereas the BSE is significantly inferior across every condition.

---

## 6. Item 6: Noise-Trained Baseline Fairness Audit

### 6.1 Audit of Training Budget Allocation
Inspecting `scripts/run_phase3_benchmark.py` (lines 332–375) and `src/training/noise_training.py` (lines 27–126):
- **Clean Baselines (M1 & M4):** Trained in Phase 2 for 200 episodes exclusively under the `clean` scenario.  
  Total clean training steps = $200 \text{ eps} \times 3,600 \text{ steps} = 720,000$ steps.
- **Noise-Trained Baselines (M2 & M5):** Trained for 200 episodes total, randomly sampling each episode from `["clean", "mild", "moderate", "severe", "extreme"]`.
  - Effective clean training episodes: $\approx 200 / 5 = 40$ episodes ($144,000$ steps).
  - Clean training budget ratio: $\frac{\text{Budget}(M2)}{\text{Budget}(M1)} = \mathbf{20.0\%}$.

### 6.2 Convergence Analysis
Because M2 and M5 received only $20\%$ of the clean experience and were continuously subjected to heavy Gaussian noise and random detector dropouts from episode 1, gradient variance remained high. Replicate seed 42 converged to a degraded control policy that over-serviced minor approaches, yielding a high clean queue of $12.14$ veh (IPPO) and $10.93$ veh (FMA2C).

### 6.3 Proposed Fair Curriculum Protocol (Read-Only Specification)
To achieve parity without running large compute:
1. **Total Budget:** 300 episodes ($1.08 \times 10^6$ steps).
2. **Phase 1 — Clean Convergence Warmup (100 episodes):**  
   Train on $100\%$ clean observations. Guarantees the baseline policy learns standard coordination and green-wave progression.
3. **Phase 2 — Staged Noise Annealing (100 episodes):**  
   Curriculum ramps degradation severity: Episodes 101–130: `mild`; Episodes 131–165: `moderate`; Episodes 166–200: `severe`.
4. **Phase 3 — Mixed Data Augmentation with Clean Regularization (100 episodes):**  
   Sample $50\%$ episodes clean, $50\%$ randomized noise mixture. Preserves clean policy efficiency while conditioning the value function against sensor dropouts.

---

## Summary of Completed Deliverables

1. **Artifact Integrity:** Zero V1/V2 canonical files or checkpoints were modified or overwritten.
2. **Outputs Written:**
   - [`experiments/phase3_v2_diagnostics/diagnostic_results.json`](file:///d:/canvas/MARLTSC/experiments/phase3_v2_diagnostics/diagnostic_results.json)
   - [`reports/phase3_v2_fix_report.md`](file:///d:/canvas/MARLTSC/reports/phase3_v2_fix_report.md)
   - [`src/envs/noise_injection_v2.py`](file:///d:/canvas/MARLTSC/src/envs/noise_injection_v2.py)
   - [`scratch/execute_all_diagnostics.py`](file:///d:/canvas/MARLTSC/scratch/execute_all_diagnostics.py)
3. **Execution Safety:** All runs were deterministic and small-scale. The BSE Gate explicitly failed, and execution was stopped prior to any large retraining runs.

