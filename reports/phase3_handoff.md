# Phase 2 to Phase 3 Handoff Specification

**Date**: November 6, 2026  
**Author**: Antigravity Quality Assurance & Verification Agent  
**Destination**: Phase 3 (Belief State Estimator & Uncertainty-Aware Integration)  
**Status**: FORMALLY APPROVED FOR HANDOFF  

---

## 1. Executive Summary

Phase 2 established the empirical foundation of the MARLTSC research program by benchmarking classical, flat MARL, constrained MARL, and hierarchical MARL controllers across observation noise sweeps on a 16-intersection grid.

This document specifies the exact empirical baseline, algorithmic priorities, architectural artifacts, and methodological requirements that Phase 3 must inherit and address.

---

## 2. Core Phase 2 Empirical Findings for Phase 3

### 2.1 The Reference Advantage
Under clean reference observations:
- **IPPO** ($5.80 \pm 0.04$ veh) and **FMA2C** ($5.82 \pm 0.10$ veh) achieve the lowest queue lengths in the benchmark.
- Both outperform calibrated **Actuated** control ($6.45 \pm 0.01$ veh) by approximately **$10.1\%$** ($p < 0.01$, paired $t$-test).
- Both dramatically outperform open-loop **Fixed-Time** control ($11.29 \pm 0.03$ veh) by **$48.6\%$**.

### 2.2 The Fragility Problem (The Core Motivation for Phase 3 BSE)
Under sensor observation noise ($\sigma \in \{0.1, 0.2, 0.3, 0.5\}$):
- **Actuated** control degrades gracefully ($\text{slope } \beta = +1.06$).
- **IPPO** degrades nearly 3x faster ($\text{slope } \beta = +2.95$), rising from $5.80 \to 7.24$ veh.
- **FMA2C** degrades similarly ($\text{slope } \beta = +2.78$), rising from $5.82 \to 7.17$ veh.
- **The Cross-Over Point**: At severe noise ($\sigma = 0.3$), Actuated ($6.85$ veh) matches or outperforms IPPO ($6.91$ veh) and FMA2C ($6.84$ veh). At extreme noise ($\sigma = 0.5$), Actuated ($6.97$ veh) beats all learned methods.

**Primary Phase 3 Goal**: Prove that the Belief State Estimator (BSE) restores RL queue performance under noise, bending the degradation slope back toward or below Actuated control ($\beta \le 1.0$).

---

## 3. Recommended Baseline Targets for BSE Integration

### 3.1 Primary Target: IPPO + BSE
- **Rationale**: IPPO is the cleanest, most performant flat MARL baseline. It uses parameter-shared actor and critic networks, conditions on local observations, and achieves the lowest clean queue ($5.80$ veh).
- Integrating BSE directly into IPPO's observation pipeline provides an unambiguous test of whether latent belief reconstruction protects decentralized policies against observation corruption without confounding hierarchical dynamics.

### 3.2 Secondary Target: FMA2C + BSE
- **Rationale**: FMA2C represents hierarchical spatial coordination across 4 regions.
- Its degradation under noise ($\beta = +2.78$) demonstrates that hierarchical abstractions fail when worker agents pass corrupted regional summaries to managers. Integrating BSE tests whether belief estimation stabilizes multi-level coordination.

---

## 4. Mandatory Noise-Trained Baseline Experiment (Phase 3 Requirement)

Phase 2 evaluated models trained exclusively on clean observations. A reviewer will rightfully ask:
> *"Does BSE provide any benefit over simply training the baseline with observation noise (data augmentation)?"*

To address this, Phase 3 **must** include the following 6-way controlled experiment under matched evaluation conditions:

| Method ID | Model Architecture | Training Condition | Evaluation Condition | Purpose |
| :---: | :--- | :--- | :--- | :--- |
| **M1** | IPPO (Clean-Trained) | Clean ($\sigma = 0.0$) | All Noise Sweeps | Baseline fragility reference |
| **M2** | IPPO (Noise-Trained) | Multi-Noise Mixture ($\sigma \in [0, 0.5]$) | All Noise Sweeps | Direct noise-robustness baseline (data augmentation) |
| **M3** | IPPO + BSE (Ours) | Clean-Trained + Online BSE | All Noise Sweeps | Test of latent belief reconstruction vs. noise-training |
| **M4** | FMA2C (Clean-Trained) | Clean ($\sigma = 0.0$) | All Noise Sweeps | Hierarchical fragility reference |
| **M5** | FMA2C (Noise-Trained) | Multi-Noise Mixture ($\sigma \in [0, 0.5]$) | All Noise Sweeps | Hierarchical noise-training baseline |
| **M6** | FMA2C + BSE (Ours) | Clean-Trained + Online BSE | All Noise Sweeps | Test of hierarchical belief stabilization |

All 6 models must be evaluated on the **exact same 10 matched traffic seeds** across the 5 canonical noise scenarios with identical 3,600-step simulation horizons.

---

## 5. Unresolved Issues and Limitations Inherited from Phase 2

1. **MAPPO-LCE Multiplier Dormancy**:
   - In the canonical Phase 2 V2 run, $\lambda_{\text{starve}} = 0.0$ across all 200 episodes in all 3 seeds.
   - The queue performance of MAPPO-LCE ($6.37$ veh) was not caused by active constraint penalties.
   - No matched $\lambda=0$ ablation exists.
   - Phase 3 must treat MAPPO-LCE constraint satisfaction as an unvalidated exploratory baseline.
2. **The 38-Second Starvation Threshold**:
   - The threshold sits between Fixed-Time ($36$s) and greedy MAPPO ($53$s).
   - It represents an internal diagnostic tuning choice, not an externally standardized safety limit.
3. **Statistical Power with $K=3$**:
   - Wilcoxon cannot achieve $p < 0.05$ with $K=3$ ($p_{\min} = 0.25$).
   - In Phase 3, the final comparison should expand independent training seeds to $K \ge 5$ (ideally $K=10$) to enable rigorous non-parametric hypothesis testing.

---

## 6. Checkpoint and Artifact Inventory for Phase 3

All pre-trained Phase 2 V2 model checkpoints are saved in `experiments/phase2_canonical_v2/models/`:

- **IPPO**: `ippo.pt`, `ippo_s43.pt`, `ippo_s44.pt`
- **FMA2C**: `fma2c.pt`, `fma2c_s43.pt`, `fma2c_s44.pt`
- **MAPPO**: `mappo.pt`, `mappo_s43.pt`, `mappo_s44.pt`
- **MAPPO-LCE**: `mappo_lce.pt`, `mappo_lce_s43.pt`, `mappo_lce_s44.pt`

Configuration snapshots are codified in `experiment_config.py` under the `'canonical'` profile ($16$ intersections, $10$ seeds, $3,600$ steps).

