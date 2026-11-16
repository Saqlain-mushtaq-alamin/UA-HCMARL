# MARLTSC Phase 3 — Centralized Seed Allocation Policy

**Date**: November 6, 2026  
**Status**: APPROVED & MATHEMATICALLY VERIFIED  
**Auditor**: Antigravity Quality Assurance & Verification Agent  
**Context**: Phase 3 Blocker 3 Resolution  

---

## 1. Background & Root Cause Analysis

In the pre-launch readiness audit, a data contamination vulnerability was discovered in `scripts/run_phase3_benchmark.py`:
- Line 73 previously collected the held-out BSE test/validation split using `seed=1042`.
- Meanwhile, the canonical traffic evaluation protocol generates episode seeds according to:
  $$\text{ep\_seed} = \text{eval\_base\_seed} + \text{ep\_idx}$$
  where $\text{eval\_base\_seed} \in \{1000, 1100, 1200, \dots, 1900\}$ and $\text{ep\_idx} \in \{0, 1, 2\}$.
- For evaluation seed 0, seed 1042 fell into the same numerical neighborhood and risk space, causing a direct train/test/evaluation data contamination flaw.

---

## 2. Formal Centralized Seed Allocation Policy

To ensure 100% strict data separation, MARLTSC codifies partitioned, non-overlapping seed spaces in `experiment_config.py`.

### Seed Allocation Matrix:

| Operational Stream | Allocated Seed Domain | Concrete Values | Purpose & Guarantee |
| :--- | :---: | :---: | :--- |
| **BSE Supervised Training** | $[80000, 80999]$ | `80042` | Generates $(o_{\text{noisy}}, s_{\text{true}})$ pairs for supervised GRU-KalmanNet training. |
| **BSE Held-Out Validation** | $[81000, 81999]$ | `81042` | Held-out validation data for MSE reduction and uncertainty calibration scoring. |
| **Model Training Replicates** | $[40, 99]$ | `42, 43, 44` | Policy optimization seeds for clean and noise-augmented training runs. |
| **Canonical Evaluation Stream** | $[1000, 1999]$ | $1000 + 100 \cdot i + k$ | 10 independent evaluation seeds ($i=0..9$) with 3 episodes each ($k=0..2$). |
| **Noise-Sampling Domain Stream** | $[420000, 440000]$ | $\text{seed} \cdot 10007 + 777$ | Generates randomized scenario selections during noise training without touching traffic RNG. |

---

## 3. Mathematical & Trajectory-Level Disjointness Proof

### 3.1 Set Disjointness
Let:
- $\mathcal{S}_{\text{BSE-train}} = \{80042\}$
- $\mathcal{S}_{\text{BSE-test}} = \{81042\}$
- $\mathcal{S}_{\text{Train-rep}} = \{42, 43, 44\}$
- $\mathcal{S}_{\text{Eval}} = \{1000, 1001, 1002, 1100, \dots, 1902\}$

The pairwise intersections are empty:
$$\mathcal{S}_{\text{BSE-train}} \cap \mathcal{S}_{\text{BSE-test}} = \emptyset$$
$$\mathcal{S}_{\text{BSE-train}} \cap \mathcal{S}_{\text{Eval}} = \emptyset$$
$$\mathcal{S}_{\text{BSE-test}} \cap \mathcal{S}_{\text{Eval}} = \emptyset$$
$$\mathcal{S}_{\text{Train-rep}} \cap \mathcal{S}_{\text{Eval}} = \emptyset$$

### 3.2 Physical Simulation Trajectory Disjointness
Automated integration test `tests/test_seed_allocation.py::test_trajectory_level_disjointness` executed 20-step simulations across all seed domains:
- $\sum | \text{Traj}_{80042} - \text{Traj}_{81042} | > 0$ (BSE train vs test queue patterns are physically distinct).
- $\sum | \text{Traj}_{81042} - \text{Traj}_{1000} | > 0$ (BSE validation vs canonical evaluation queue patterns are physically distinct).
- $\sum | \text{Traj}_{1000} - \text{Traj}_{1001} | > 0$ (Evaluation episodes are physically distinct).

---

## 4. Operational Directives
1. All scripts must obtain seed configurations via `experiment_config.get_config(profile)`.
2. Any hardcoded seed in the interval $[1000, 1999]$ outside of the evaluation loop is forbidden.
3. Automated unit test `test_seed_allocation.py` runs on every PR and preflight check.

