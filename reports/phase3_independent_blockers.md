# Phase 3 Independent Blocker & Defect Log

**Auditor Role**: Independent Senior ML Systems & Experimental Methodology Auditor  
**Date**: November 6, 2026  
**Context**: Pre-Launch Canonical Verification  
**Status**: ACTIVE BLOCKERS REQUIRING REMEDIATION  

---

## 1. Overview of Audit Blockers

The independent audit identified four critical defects that prevent Phase 3 from producing scientifically valid, publication-grade results.

| Blocker ID | Severity | Primary Component | Impact Area | Verdict |
| :---: | :---: | :--- | :--- | :---: |
| **AUDIT-B1** | **CRITICAL** | `scripts/run_phase3_benchmark.py` | Statistical Validity / Multi-Seed Replicates | **FAIL** |
| **AUDIT-B2** | **CRITICAL** | `src/baselines/bse_wrapper.py:143` | Mathematical Integrity of BSE Kalman Recurrence | **FAIL** |
| **AUDIT-B3** | **HIGH** | `scripts/run_phase3_benchmark.py:192, 217` | State Pollution via Shared Controller Instances | **FAIL** |
| **AUDIT-B4** | **MEDIUM** | `scripts/run_phase3_benchmark.py:365` | Reproducibility & Cryptographic Provenance | **FAIL** |

---

## 2. Exhaustive Blocker Analysis & Corrective Actions

### 2.1 AUDIT-B1: Single Replicate Execution & Statistical Pseudo-Replication

- **Verdict**: **FAIL**
- **Exact File & Location**: `scripts/run_phase3_benchmark.py`, lines 170, 184, 202, 214, 317–321.
- **Evidence from Code**:
  ```python
  # Line 170:
  m1_ippo_clean = get_clean_controller("ippo", seed=42)
  # Line 184:
  train_noise_augmented_controller(controller=m2_ippo_noise, ..., seed=42)
  # Line 202:
  m4_fma2c_clean = get_clean_controller("fma2c", seed=42)
  # Line 214:
  train_noise_augmented_controller(controller=m5_fma2c_noise, ..., seed=42)
  ```
  And in lines 317–320:
  ```python
  clean_delays = {m: results["clean"][m]["avg_delay"] for m in arms.keys()}
  stat_report = generate_statistical_report(clean_delays, baseline_name="M1: IPPO (Clean)")
  ```
- **Why It Matters**:
  1. `cfg.training_seeds` in `experiment_config.py` is `[42, 43, 44]`. Phase 2 pre-trained and saved models for all 3 seeds: `ippo.pt`, `ippo_s43.pt`, `ippo_s44.pt`, `fma2c.pt`, `fma2c_s43.pt`, `fma2c_s44.pt`.
  2. The Phase 3 benchmark runner only instantiates and evaluates seed 42 ($K=1$).
  3. `results["clean"][m]["avg_delay"]` is a list of 10 evaluation traffic episodes from a *single* model checkpoint.
  4. Running Student's $t$-tests and Wilcoxon signed-rank tests across evaluation traffic episodes from a single model treats traffic variation as independent policy replicates. This is the exact statistical pseudo-replication error corrected in Phase 2.
  5. The true unit of statistical independence is the **training seed replicate** ($K=3$).
- **Required Corrective Action**:
  - Refactor `run_phase3_benchmark.py` to evaluate across all `training_seeds` in `cfg.training_seeds` (`[42, 43, 44]`).
  - For each training seed $s \in \{42, 43, 44\}$:
    - Load/train the corresponding controller replicate $s$.
    - Evaluate across the canonical traffic seed stream.
    - Compute the replicate-level mean metric for each seed.
  - Compute hypothesis tests across the $K=3$ independent replicate means.

---

### 2.2 AUDIT-B2: Missing Kalman Recurrence in BSE Wrapper

- **Verdict**: **FAIL**
- **Exact File & Location**: `src/baselines/bse_wrapper.py`, lines 141–152.
- **Evidence from Code**:
  ```python
  # bse_wrapper.py lines 141-145:
  self.bse.eval()
  with torch.no_grad():
      b_hat_t, sigma_hat_t, _ = self.bse(hist_batch_t)
      b_hat_np = b_hat_t.cpu().numpy()
      sigma_np = sigma_hat_t.squeeze(-1).cpu().numpy()
  ```
  Compare with `src/models/belief_encoder.py:98-103`:
  ```python
  # Prediction-Correction Update
  if prev_belief is None:
      prev_belief = current_obs
  predicted_belief = self.transition_net(prev_belief)
  b_hat = (1.0 - kalman_k) * predicted_belief + kalman_k * current_obs
  ```
- **Why It Matters**:
  1. `self.prev_beliefs` is maintained in `__init__` and `reset()`, but **never passed** to `self.bse()` in `select_actions()`, and **never updated** with `b_hat_t`.
  2. Because `prev_belief` is always `None`, `GRUKalmanNet.forward()` falls back to `prev_belief = current_obs` on *every single simulation step*.
  3. Consequently, the transition network $\hat{\mathbf{b}}_t^- = f(o_t)$ predicts from the current noisy observation $o_t$ rather than the prior belief $\hat{\mathbf{b}}_{t-1}$.
  4. The model does not execute the claimed Kalman recursive update:
     $$\hat{\mathbf{b}}_t = (1 - \mathbf{K}_t) f(\hat{\mathbf{b}}_{t-1}) + \mathbf{K}_t \mathbf{o}_t$$
     Instead, it runs an un-recurrent feedforward residual filter:
     $$\hat{\mathbf{b}}_t = (1 - \mathbf{K}_t) f(\mathbf{o}_t) + \mathbf{K}_t \mathbf{o}_t$$
  5. This violates the core theoretical novelty of Layer 1 published in the blueprint.
- **Required Corrective Action**:
  - In `bse_wrapper.py:126-146`:
    - Construct the batched `prev_belief_batch_t` tensor from `self.prev_beliefs`.
    - Pass `prev_belief=prev_belief_batch_t` to `self.bse(hist_batch_t, prev_belief=prev_belief_batch_t)`.
    - Update `self.prev_beliefs[aid] = b_hat_t[idx].detach()` after each forward pass.
  - Also verify that `src/training/bse_trainer.py` pre-trains `GRUKalmanNet` on sequential rollouts with belief recursion or documents that supervised pre-training initializes with windowed unrolling.

---

### 2.3 AUDIT-B3: State Pollution via Shared In-Memory Controller Instances

- **Verdict**: **FAIL**
- **Exact File & Location**: `scripts/run_phase3_benchmark.py`, lines 170, 192, 202, 217.
- **Evidence from Code**:
  ```python
  # Line 170 & 192:
  m1_ippo_clean = get_clean_controller("ippo", seed=42)
  m3_ippo_bse = BSEControllerWrapper(base_controller=m1_ippo_clean, bse_model=bse_model, history_len=5)

  # Line 202 & 217:
  m4_fma2c_clean = get_clean_controller("fma2c", seed=42)
  m6_fma2c_bse = BSEControllerWrapper(base_controller=m4_fma2c_clean, bse_model=bse_model, history_len=5)
  ```
- **Why It Matters**:
  1. `m3_ippo_bse.base_controller` is the exact same object reference as `m1_ippo_clean`.
  2. `m6_fma2c_bse.base_controller` is the exact same object reference as `m4_fma2c_clean`.
  3. While `IPPOController` is stateless during `evaluate=True`, `FMA2CController` mutates internal attributes during evaluation:
     `src/baselines/fma2c.py:195`: `self.current_goals[r_idx] = goal_np`.
  4. Evaluating Arm M4 followed by Arm M6 on the same Python instance within the same simulation step or episode loop causes sub-goal state pollution: residual sub-goals from M4 remain in memory when M6 begins, and vice versa.
  5. In-place resetting of M3 also mutates M1.
- **Required Corrective Action**:
  - Instantiate separate, independent controller instances for M3 and M6:
    ```python
    m1_ippo_clean = get_clean_controller("ippo", seed=42)
    m3_ippo_clean_base = get_clean_controller("ippo", seed=42)
    m3_ippo_bse = BSEControllerWrapper(base_controller=m3_ippo_clean_base, bse_model=bse_model, history_len=5)

    m4_fma2c_clean = get_clean_controller("fma2c", seed=42)
    m6_fma2c_clean_base = get_clean_controller("fma2c", seed=42)
    m6_fma2c_bse = BSEControllerWrapper(base_controller=m6_fma2c_clean_base, bse_model=bse_model, history_len=5)
    ```

---

### 2.4 AUDIT-B4: Missing Cryptographic Hashes & Provenance in Experiment Manifest

- **Verdict**: **FAIL**
- **Exact File & Location**: `scripts/run_phase3_benchmark.py`, lines 365–385; `experiments/phase3_smoke_test/phase3_manifest.json`.
- **Evidence from Code**:
  ```python
  manifest = {
      "timestamp": datetime.now().isoformat(),
      "profile": profile,
      "config": cfg.to_dict(),
      "status": "COMPLETED",
      "arms_evaluated": list(arms.keys()),
      "output_files": [
          "phase3_raw_results.json",
          "bse_calibration_metrics.json", ...
      ],
      "calibration": calib_metrics,
  }
  ```
- **Why It Matters**:
  1. The manifest records filenames as simple strings without computing cryptographic checksums (SHA-256).
  2. If an experiment output file is overwritten, modified, or partially corrupted after execution, the manifest cannot detect tampering or file modification.
  3. Git commit SHA and branch name are not recorded, compromising reproducibility across machines or repository states.
  4. Task 7 explicitly mandates: *"The manifest hashes the relevant artifacts."*
- **Required Corrective Action**:
  - Implement a helper that computes SHA-256 hashes of all output files and model checkpoints:
    ```python
    def compute_file_hash(filepath: str) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()
    ```
  - Record git commit SHA via `git rev-parse HEAD`.
  - Embed artifact hashes into `phase3_manifest.json`.

---

## 3. Remediation Roadmap

To transition Phase 3 from **NO-GO** to **GO**:
1. Patch `src/baselines/bse_wrapper.py` to maintain and pass `prev_belief` during batched inference (Resolves AUDIT-B2).
2. Instantiate independent base controller instances for Arms M3 and M6 (Resolves AUDIT-B3).
3. Update `scripts/run_phase3_benchmark.py` to loop over all 3 training seeds (`42, 43, 44`) for all 6 arms, computing statistics across replicate means (Resolves AUDIT-B1).
4. Integrate SHA-256 checksums and git commit SHA into `phase3_manifest.json` (Resolves AUDIT-B4).
5. Re-run regression tests and verify clean smoke execution.

