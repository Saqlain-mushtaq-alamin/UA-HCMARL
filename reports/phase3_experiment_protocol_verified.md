# Phase 3 Verified Experiment Protocol Specification

**Date**: November 6, 2026  
**Document Type**: Canonical Experiment Protocol Specification  
**Destination**: Phase 3 Execution Engine  
**Status**: FORMALLY SPECIFIED  

---

## 1. Experimental Objective & Hypotheses

Phase 3 evaluates the **Belief State Estimator (BSE)** (Layer 1 GRU-KalmanNet) as a robust observation preprocessor for multi-agent traffic signal control.

### Primary Scientific Hypotheses:
1. **Hypothesis 1 (Fragility Mitigation)**: Clean-trained flat MARL (IPPO) and hierarchical MARL (FMA2C) suffer severe queue degradation under sensor noise ($\beta \ge 2.7$). Coupling clean-trained policies with BSE will bend the degradation slope back toward or below calibrated Actuated control ($\beta_{\text{BSE}} \le 1.0$).
2. **Hypothesis 2 (BSE vs. Data Augmentation)**: While noise-trained baselines (training with observation noise) improve noise robustness, BSE provides equal or superior noise robustness **without sacrificing clean reference performance**, whereas noise-trained baselines suffer a clean performance penalty (the "noise-regularization tax").
3. **Hypothesis 3 (Hierarchical Stabilization)**: In hierarchical control (FMA2C), observation noise corrupts both regional manager sub-goals and local worker actions. Denoising through BSE protects the hierarchical abstraction layer.

---

## 2. Six-Arm Controlled Experimental Matrix

All six arms are evaluated on the exact same 16-intersection grid under identical simulation horizons and traffic arrival seeds.

| Arm ID | Algorithmic Label | Base Architecture | Observation Input | Training Regime | Training Budget | Checkpoint Lineage |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | IPPO (Clean-Trained) | Shared IPPO Actor/Critic | Raw Noisy Obs $\mathbf{o}_t \in \mathbb{R}^{16}$ | Clean ($\sigma = 0.0$) | 200 eps $\times$ 3,600s | Inherited from Phase 2 V2 (`ippo.pt`) |
| **M2** | IPPO (Noise-Trained) | Shared IPPO Actor/Critic | Raw Noisy Obs $\mathbf{o}_t \in \mathbb{R}^{16}$ | Multi-Noise Mixture ($\sigma \in [0, 0.5]$) | 200 eps $\times$ 3,600s | Trained fresh in Phase 3 (`ippo_noise.pt`) |
| **M3** | IPPO + BSE (Ours) | Shared IPPO Actor/Critic | Denoised Belief $\hat{\mathbf{b}}_t \in \mathbb{R}^{16}$ | Clean Base + Supervised BSE | 200 eps clean + 20 ep BSE | Inherited Base (`ippo.pt`) + Trained BSE (`bse.pt`) |
| **M4** | FMA2C (Clean-Trained) | 4 Managers + 1 Shared Worker | Raw Noisy Obs $\mathbf{o}_t \in \mathbb{R}^{16}$ | Clean ($\sigma = 0.0$) | 200 eps $\times$ 3,600s | Inherited from Phase 2 V2 (`fma2c.pt`) |
| **M5** | FMA2C (Noise-Trained) | 4 Managers + 1 Shared Worker | Raw Noisy Obs $\mathbf{o}_t \in \mathbb{R}^{16}$ | Multi-Noise Mixture ($\sigma \in [0, 0.5]$) | 200 eps $\times$ 3,600s | Trained fresh in Phase 3 (`fma2c_noise.pt`) |
| **M6** | FMA2C + BSE (Ours) | 4 Managers + 1 Shared Worker | Denoised Belief $\hat{\mathbf{b}}_t \in \mathbb{R}^{16}$ | Clean Base + Supervised BSE | 200 eps clean + 20 ep BSE | Inherited Base (`fma2c.pt`) + Trained BSE (`bse.pt`) |

---

## 3. Environmental and Simulation Configuration

- **Simulation Grid**: 16 intersections ($4 \times 4$ grid, `networks/grid4x4/grid4x4.net.xml`).
- **Demand Profile**: Uniform arterial traffic demand (`networks/grid4x4/demand_uniform.rou.xml`).
- **Step Duration**: 1.0 second per simulation step.
- **Episode Horizon**: 3,600 simulation steps per episode (1.0 wall-clock hour of simulated traffic).
- **Phases per Intersection**:
  - Phase 0: North-South Green (Straight/Right)
  - Phase 1: North-South Clearance Yellow (3 seconds)
  - Phase 2: East-West Green (Straight/Right)
  - Phase 3: East-West Clearance Yellow (3 seconds)
- **Constraints**:
  - Minimum Green: 5 seconds.
  - Maximum Green: 50 seconds.
  - Starvation / Maximum Red Wait: 38 seconds.
- **Action Masking**: Strictly enforced during minimum green and yellow clearance intervals.

---

## 4. Observation Degradation Protocol (Novelty N5)

Five standardized observation noise scenarios are evaluated:

| Scenario Preset | Gaussian Noise ($D_1$) Scale $\sigma$ | Random Dropout ($D_2$) Rate $p_{\text{drop}}$ | Systematic Bias ($D_3$) Factor $b$ |
| :---: | :---: | :---: | :---: |
| **Clean** | $0.00$ | $0.00$ | $0.00$ |
| **Mild** | $0.10$ | $0.05$ | $0.00$ |
| **Moderate** | $0.20$ | $0.10$ | $0.10$ |
| **Severe** | $0.30$ | $0.20$ | $0.20$ |
| **Extreme** | $0.50$ | $0.30$ | $0.30$ |

---

## 5. Seed Allocation and Statistical Independence

To guarantee zero data contamination and rigorous statistical power:

### 5.1 Training Replicates
- Independent training replicate seeds: $K = 3$ seeds (`42, 43, 44`).
- Each replicate represents a full independent policy optimization from different weight initializations and environment arrival seeds.

### 5.2 BSE Dataset Generation Seeds (Isolated Space)
- BSE Supervised Training Data: `seed = 80042` (2,000 steps).
- BSE Held-out Validation Data: `seed = 81042` (500 steps).
- These seeds are completely disjoint from both training and evaluation seeds.

### 5.3 Canonical Evaluation Seeds
- Evaluation seeds: 10 fixed traffic arrival seeds (`1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900`).
- Evaluation episodes per seed: 3 independent runs.
- Total evaluation episodes per scenario per replicate: $10 \times 3 = 30$ episodes.
- Across 5 noise scenarios: $30 \times 5 = 150$ episodes per replicate.
- Total evaluation workload: $6 \text{ methods} \times 3 \text{ replicates} \times 150 \text{ episodes} = 2,700$ canonical episodes.

---

## 6. Primary Evaluation Metrics & Mathematical Definitions

1. **Mean Approach Queue Length ($Q$, veh)**:
   $$Q = \frac{1}{T} \sum_{t=1}^T \frac{1}{4N} \sum_{i=1}^N \sum_{a=1}^4 q_{i,a}(t)$$
   Primary operational metric measuring traffic accumulation across the network.
2. **Robustness Degradation Slope ($\beta$, veh/noise-unit)**:
   $$\beta = \frac{\sum_{s=1}^5 (\sigma_s - \bar{\sigma})(Q_s - \bar{Q})}{\sum_{s=1}^5 (\sigma_s - \bar{\sigma})^2}$$
   where $\sigma \in \{0.0, 0.1, 0.2, 0.3, 0.5\}$. A smaller or flat slope indicates high robustness against sensor corruption.
3. **Starvation Violation Rate ($V_{\text{starve}}$, %)**:
   Percentage of simulation steps where at least one queued approach experiences red wait $\ge 38$ seconds.
4. **Network Throughput ($\mathcal{T}$, veh/hr)**:
   Total vehicles completing intersection clearance per hour of simulation.
5. **BSE Denoising Mean Squared Error ($\text{MSE}_{\text{BSE}}$)**:
   $$\text{MSE}_{\text{BSE}} = \frac{1}{N_{\text{samples}}} \sum_{k=1}^{N_{\text{samples}}} \|\hat{\mathbf{b}}_k - \mathbf{s}_k^{\text{true}}\|^2$$
6. **Uncertainty Calibration Correlation ($r(\hat{\sigma}, \epsilon)$)**:
   Pearson correlation between predicted uncertainty score $\hat{\sigma}_t$ and empirical estimation error $\epsilon_t = \|\hat{\mathbf{b}}_t - \mathbf{s}_t^{\text{true}}\|_1$.

---

## 7. Statistical Hypothesis Testing Protocol

- **Independent Unit of Analysis**: Training seed replicates ($K = 3$).
- **Paired Comparisons**:
  - M1 (Clean IPPO) vs. M3 (IPPO + BSE) [Paired across identical evaluation seeds]
  - M2 (Noise-Trained IPPO) vs. M3 (IPPO + BSE)
  - M4 (Clean FMA2C) vs. M6 (FMA2C + BSE)
  - M5 (Noise-Trained FMA2C) vs. M6 (FMA2C + BSE)
- **Statistical Tests**:
  - Two-tailed paired Student's $t$-test.
  - Non-parametric Wilcoxon signed-rank test.
  - 95% Confidence Intervals computed via bootstrapping over evaluation seeds.

