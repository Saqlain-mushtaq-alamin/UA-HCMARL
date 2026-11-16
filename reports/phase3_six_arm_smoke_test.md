# MARLTSC Phase 3 — Six-Arm Smoke Test Report

**Date**: November 6, 2026  
**Execution Environment**: Python 3.11.9, PyTorch 2.6.0+cpu, Gymnasium 0.29.1, PettingZoo 1.24.3  
**Auditor**: Antigravity Quality Assurance & Verification Agent  
**Context**: Pre-Launch Readiness Verification (Phase 3 Blocker 4 Validation)  
**Profile**: `smoke` (4 intersections, 50 steps/episode, 2 eval seeds, 1 episode/seed)  
**Output Directory**: `experiments/phase3_smoke_test/`  
**Status**: **PASSED (ALL 6 ARMS VERIFIED END-TO-END)**  

---

## 1. Executive Summary

A non-destructive, isolated smoke test was executed across all six experimental arms using the refactored canonical runner `scripts/run_phase3_benchmark.py`. 

The test verified:
1. **Layer 1 GRU-KalmanNet Pre-Training**: Successful supervised training on seed 80042 and validation on seed 81042.
2. **Controller Instantiation**: All six controller arms (M1–M6) initialized, loaded, and stepped without tensor dimension or device mismatches.
3. **BSE Observation Wrapping**: `BSEControllerWrapper` batched inference over all intersections and delivered denoised states to IPPO and FMA2C without memory leaks or parameter updates.
4. **Noise Injection**: Environment observation noise was actively injected across Clean, Mild, Moderate, Severe, and Extreme presets.
5. **Action Masking**: Minimum-green and clearance intervals enforced valid actions throughout.
6. **Artifact Generation**: Full set of artifacts, checkpoint files, statistical summaries, figures, and machine-readable experiment manifest were written atomically to disk.

---

## 2. Six-Arm Evaluation Results (Smoke Profile)

| Arm ID | Algorithmic Identifier | Model Architecture | Training Regime | Queue Length (veh) | Starvation Rate (%) |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **M1** | IPPO (Clean) | Parameter-Shared Actor/Critic | Clean ($\sigma = 0.0$) | $8.49 \pm 2.0$ | $9.7\%$ |
| **M2** | IPPO (Noise-Trained) | Parameter-Shared Actor/Critic | Multi-Noise Mixture | $6.21 \pm 2.3$ | $5.0\%$ |
| **M3** | IPPO + BSE (Ours) | Clean IPPO + GRU-KalmanNet | Clean Base + BSE | $8.81 \pm 2.0$ | $12.2\%$ |
| **M4** | FMA2C (Clean) | 2 Regional Mgrs + Shared Worker | Clean ($\sigma = 0.0$) | $5.74 \pm 0.2$ | $0.0\%$ |
| **M5** | FMA2C (Noise-Trained) | 2 Regional Mgrs + Shared Worker | Multi-Noise Mixture | $5.68 \pm 0.3$ | $0.0\%$ |
| **M6** | FMA2C + BSE (Ours) | Clean FMA2C + GRU-KalmanNet | Clean Base + BSE | $5.51 \pm 0.0$ | $0.0\%$ |

*Note: Smoke profile uses only 2 training episodes and 50 steps for validation of execution plumbing; quantitative values do not represent canonical convergence.*

---

## 3. Verified Artifact Inventory

All required outputs were verified in `experiments/phase3_smoke_test/`:
- `phase3_raw_results.json`: Full records of delays, violation rates, throughput, and BSE latency.
- `bse_calibration_metrics.json`: Raw MSE ($0.1439$), BSE MSE ($0.1234$), MSE reduction ($14.2\%$).
- `phase3_statistical_summary.json`: Paired Student's $t$-test and Wilcoxon signed-rank comparisons against M1.
- `models/bse.pt`: Pre-trained GRU-KalmanNet checkpoint (99.7 KB).
- `models/ippo_noise.pt`: Multi-noise trained IPPO checkpoint (89.3 KB).
- `models/fma2c_noise.pt`: Multi-noise trained FMA2C checkpoint (192.3 KB).
- `fig1_clean_preservation.png`: Publication bar chart of clean reference performance.
- `fig2_robustness_noise_sweep.png`: Degradation curves across 5 noise levels.
- `fig3_bse_vs_noise_training.png`: Comparative robustness trajectories (M1 vs M2 vs M3).
- `fig4_bse_calibration_quality.png`: Pre-training MSE denoising comparison.
- `phase3_manifest.json`: Verified experiment manifest with timestamps, config hash, and status `COMPLETED`.

---

## 4. Diagnostic Conclusion

The six-arm benchmark runner executes cleanly from end to end without crashing, memory leaks, or unhandled exceptions. Blockers 1, 2, 3, and 4 are confirmed resolved at the runtime level.

