# Phase 3 Canonical Benchmark Launch Decision

**Document Type**: Formal Launch Authorization Verdict  
**Auditor Role**: Independent Senior ML Systems & Experimental Methodology Auditor  
**Date**: November 6, 2026  
**Subject**: MARLTSC Phase 3 (Layer 1 Belief State Estimation for Robust Multi-Agent TSC)  
**Target Execution Command**: `python scripts/run_phase3_benchmark.py --profile canonical`  

---

## 1. Formal Launch Decision

### Final Verdict: **NO-GO**

**AUTHORIZATION TO LAUNCH THE CANONICAL PHASE 3 BENCHMARK IS DENIED.**

---

## 2. Decision Rationale

Although the implementation agent successfully built modular primitives (`BSEControllerWrapper`, `noise_training.py`, centralized seed configuration) and all 63 unit tests pass, independent audit revealed **four critical blockers** that make launching the canonical experiment scientifically invalid, methodologically flawed, and premature:

1. **Statistical Pseudo-Replication (Blocker AUDIT-B1)**:
   The canonical runner evaluates only **seed 42** ($K=1$) instead of the mandated 3 independent training replicate seeds (`42, 43, 44`). Statistical significance tests in `phase3_statistical_summary.json` are performed across 10 evaluation traffic episodes of a single model, reviving the pseudo-replication defect that compromised early Phase 2 results.
2. **Mathematical Violation of Published Kalman Recurrence (Blocker AUDIT-B2)**:
   In `src/baselines/bse_wrapper.py:143`, `self.bse(hist_batch_t)` omits `prev_belief`. Consequently, the transition network $f(\cdot)$ operates on the current noisy observation $o_t$ rather than the prior belief $\hat{\mathbf{b}}_{t-1}$, reducing the published Kalman filter recurrence $\hat{\mathbf{b}}_t = (1 - \mathbf{K}_t) f(\hat{\mathbf{b}}_{t-1}) + \mathbf{K}_t \mathbf{o}_t$ to a static feedforward filter.
3. **Cross-Arm Memory Mutation & State Pollution (Blocker AUDIT-B3)**:
   Arms M3 and M6 wrap the exact same in-memory controller instances as M1 and M4. Because `FMA2CController` mutates `self.current_goals` in-place during evaluation, sequential evaluation of M4 and M6 on the same object introduces hidden state contamination.
4. **Lack of Cryptographic Artifact Provenance (Blocker AUDIT-B4)**:
   `phase3_manifest.json` fails to record SHA-256 checksums of model checkpoints and raw result files, and omits the git commit SHA, violating research provenance standards.

Launching a 2,700-episode canonical benchmark in this state would waste substantial compute resources, produce un-recurrent belief representations, and yield statistically invalid conclusions.

---

## 3. Mandatory Conditions for "GO" Authorization

The decision will be upgraded to **GO** only after the following remediation conditions are verified on disk:

1. **Patch BSE Kalman Recurrence**:
   Modify `src/baselines/bse_wrapper.py` to maintain batched `self.prev_beliefs` and pass them into `self.bse(hist_batch_t, prev_belief=prev_belief_batch)`.
2. **Decouple Base Controller Instances**:
   In `scripts/run_phase3_benchmark.py`, instantiate distinct, independent controller objects for M3 (`IPPO + BSE`) and M6 (`FMA2C + BSE`).
3. **Implement Full $K=3$ Replicate Benchmark Architecture**:
   Update `scripts/run_phase3_benchmark.py` to loop over all 3 training seeds (`[42, 43, 44]`) for all six arms, saving per-replicate models and computing paired statistical tests across replicate means.
4. **Implement Cryptographic Manifest Generation**:
   Compute SHA-256 hashes of all output files and checkpoints, log git commit revision, and embed them in `phase3_manifest.json`.
5. **Pass Re-Audit**:
   Execute a smoke test with these fixes and demonstrate clean verification across all 4 remediated areas.

---

## 4. Strict Enforcement Directive

- **DO NOT** launch `scripts/run_phase3_benchmark.py --profile canonical`.
- **DO NOT** train or evaluate full canonical workloads until authorization is upgraded to **GO**.
- Preserve Phase 2 canonical artifacts (`experiments/phase2_canonical_v2/`) unchanged.

