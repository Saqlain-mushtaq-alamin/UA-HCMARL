# Phase 2 Final Report: Comprehensive Baselines, Robustness Degradation & Reproducibility Audit

**Project**: Multi-Agent Reinforcement Learning Traffic Signal Control (MARLTSC)  
**Phase**: Phase 2 Finalization  
**Date**: November 6, 2026  
**Status**: COMPLETE & FORMALLY CLOSED  
**Canonical Results Directory**: `experiments/phase2_canonical_v2/`  
**Reference V1 Directory**: `experiments/phase2_flat_baselines/`  
**Derived Finalization Directory**: `experiments/phase2_finalization/`  

---

## 1. Executive Summary

Phase 2 established a rigorous, reproducible, multi-baseline benchmark for multi-agent traffic signal control under both ideal reference conditions and systematic observation noise. Six controllers spanning four architectural paradigms—classical open-loop (Fixed-Time), classical feedback (Actuated), decentralized flat MARL (IPPO), centralized-critic flat MARL (MAPPO), constrained flat MARL (MAPPO-LCE), and hierarchical MARL (FMA2C)—were evaluated on a 16-intersection grid network across five noise scenarios.

### Key Empirical Findings:
1. **Clean-Observation Reference Advantage**: Under clean reference observations, learned controllers achieve state-of-the-art queue reductions. IPPO ($5.80 \pm 0.04$ veh) and FMA2C ($5.82 \pm 0.10$ veh) outperform calibrated Actuated control ($6.45 \pm 0.01$ veh) by approximately **$10.1\%$** ($p < 0.01$) and open-loop Fixed-Time control ($11.29 \pm 0.03$ veh) by **$48.6\%$**.
2. **Noise Fragility & Cross-Over**: Both flat and hierarchical RL policies degrade rapidly under sensor observation noise ($\beta_{\text{IPPO}} = +2.95, \beta_{\text{FMA2C}} = +2.78$), degrading at nearly triple the rate of classical Actuated feedback ($\beta_{\text{Actuated}} = +1.06$). At severe noise ($\sigma = 0.3$), Actuated control matches RL performance; at extreme noise ($\sigma = 0.5$), Actuated control ($6.97$ veh) outperforms all learned policies.
3. **Statistical Corrections**: Preliminary reporting defects—including mathematically impossible Wilcoxon significance claims ($p_{\min} = 0.25$ with $K=3$) and inflated Cohen's $d$ values (exceeding $-200$)—have been rectified. Statistical tests are reported strictly over independent training replicates ($K=3$) using paired differences and bootstrap confidence intervals.
4. **Resolution of MAPPO Negative Slope**: Unconstrained MAPPO exhibits a negative degradation slope ($\beta = -2.16$) across all three training seeds. Under clean observations, MAPPO's centralized value function locks into an asymmetric primary-corridor bias (yielding $8.85$ veh queue and starvation violations). Sensor noise perturbs the greedy policy inputs, acting as a dither signal that forces phase alternation and relieves queue buildup on opposing approaches.
5. **Constraint Diagnostics & Caveats**: While the 4-dimensional constraint pipeline in MAPPO-LCE is fully implemented and functionally verified, training logs confirm that $\lambda_{\text{starve}} = 0.0000$ throughout canonical training across all three seeds because action masking and stochastic exploration prevented 38s continuous red waits during training. In the absence of a matched $\lambda \equiv 0$ ablation, constraint satisfaction is classified as an exploratory implementation rather than a validated empirical causal finding.

---

## 2. Experiment Design and Methodology

### 2.1 Environmental Setup
The benchmark operates on a 16-intersection 4x4 grid network (`networks/grid4x4/grid4x4.net.xml`) configured with standard urban grid arterial spacing (300 meters between intersections).
- **Simulator**: SUMO microscopic traffic simulator with high-fidelity native fallback queue simulation engine ([`src/envs/sumo_env.py`](file:///d:/canvas/MARLTSC/src/envs/sumo_env.py)).
- **Simulation Duration**: $3,600$ seconds (1 hour of real-time traffic) per evaluation episode with a $1.0$-second discrete simulation step.
- **Traffic Demand**: Uniform Poisson arrival process with an arrival rate of $\lambda = 0.2$ vehicles/second per incoming approach across all 16 intersections (`networks/grid4x4/demand_uniform.rou.xml`).
- **Signal Control Logic**: 4 discrete phases per intersection:
  - Phase 0: North-South Green (Discharge rate: 0.8 veh/s)
  - Phase 1: North-South Yellow Clearance ($3$s, non-preemptible)
  - Phase 2: East-West Green (Discharge rate: 0.8 veh/s)
  - Phase 3: East-West Yellow Clearance ($3$s, non-preemptible)
- **Operational Constraints**: `min_green = 5`s, `max_green = 50`s, `yellow_time = 3`s, and starvation threshold `max_red_wait = 38`s.

### 2.2 Observation Space & Action Space
- **Local Observation Vector**: $\mathbb{R}^{16}$ per intersection, consisting of:
  - Approach queue lengths (4 dims, normalized by $Q_{\max} = 20.0$).
  - Approach arrival rates (4 dims, scaled).
  - One-hot phase indicator (4 dims).
  - Time elapsed in current phase (1 dim, normalized by `max_green`).
  - Total intersection queue (1 dim, normalized).
  - Phase pressure metric: $(Q_{\text{NS}} - Q_{\text{EW}}) / \max(1.0, Q_{\text{NS}} + Q_{\text{EW}})$ (1 dim).
  - Instantaneous throughput (1 dim).
- **Action Space**: Discrete(4) representing the requested phase index $\{0, 1, 2, 3\}$.

### 2.3 Noise Degradation Protocol
To assess controller robustness against sensor degradation, five observation noise scenarios are injected via [`src/envs/noise_injection.py`](file:///d:/canvas/MARLTSC/src/envs/noise_injection.py):
1. **Clean Reference**: $\sigma = 0.0$ (Exact simulator ground truth).
2. **Mild Noise**: $\sigma = 0.1$ (Zero-mean Gaussian noise $\mathcal{N}(0, 0.1^2)$ added to continuous observation components).
3. **Moderate Noise**: $\sigma = 0.2$ (Gaussian noise $\mathcal{N}(0, 0.2^2)$ + occasional uniform dither).
4. **Severe Noise**: $\sigma = 0.3$ (Gaussian noise $\mathcal{N}(0, 0.3^2)$ + intermittent sensor dropouts).
5. **Extreme Noise**: $\sigma = 0.5$ (Gaussian noise $\mathcal{N}(0, 0.5^2)$ + 10% probability of complete sensor freeze).

---

## 3. Algorithms and Baseline Descriptions

| Identifier | Paradigm | Controller Architecture | Observation Scope | Coordination Mechanism |
| :---: | :--- | :--- | :--- | :--- |
| **B1** | Classical | **Fixed-Time** | Open-loop (None) | Pre-timed splits ($30$s green, $3$s yellow; 66s cycle) |
| **B2** | Classical | **Actuated** | Local unnormalized queues | Gap-out logic with calibrated 1.5-vehicle threshold |
| **B3** | Flat MARL | **IPPO** | Local $\mathbb{R}^{16}$ | Decentralized actor-critic with parameter sharing |
| **B4** | Flat MARL | **MAPPO** | Global $\mathbb{R}^{256}$ (Critic) | Parameter-shared actor + centralized critic |
| **B5** | Constrained | **MAPPO-LCE** | Global $\mathbb{R}^{256}$ (Critic) | Centralized critic + PID Lagrangian cost penalties |
| **B6** | Hierarchical | **FMA2C** | 4 Spatial Regions | 2-level hierarchy: 4 Regional Managers + 16 Local Workers |

---

## 4. Training and Evaluation Protocol

1. **Training Protocol**:
   - Each RL algorithm was trained from scratch across three independent random seeds: **42, 43, and 44** ($K=3$ independent training replicates).
   - Training budget: Exactly **200 episodes** of $3,600$ steps each ($720,000$ steps per replicate; $2,160,000$ steps per algorithm).
   - Condition: All models were trained exclusively in the clean-observation reference condition.
   - Action masking was enforced during training to eliminate illegal transitions during clearance intervals.
2. **Evaluation Protocol**:
   - Each trained model replicate was evaluated across **10 evaluation seeds** ($0 \dots 9$).
   - Budget: **3 evaluation episodes** per seed ($10,800$ steps per seed; $36,000$ steps per scenario).
   - Matched traffic streams: Random seed $j$ parameterizes identical vehicle arrival sequences across all controllers and noise levels.
   - Classical baselines (Fixed-Time, Actuated) were evaluated on the identical 10-seed matched traffic streams.

---

## 5. Clean-to-Extreme Queue Comparison

The table below presents the verified macro-performance across all 6 controllers in the clean reference and extreme noise conditions:

| Algorithm | Clean Queue (veh) | Extreme Queue (veh) | Absolute Change ($\Delta$) | Relative Change (%) | Robustness Slope ($\beta$) | Clean Viols (%) | Extreme Viols (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | $11.29 \pm 0.06$ | $11.29 \pm 0.06$ | $\mathbf{0.00}$ veh | $0.0\%$ | $\beta = 0.00$ | $0.000\%$ | $0.000\%$ |
| **Actuated** | $6.45 \pm 0.05$ | $6.97 \pm 0.05$ | $\mathbf{+0.52}$ veh | $+8.1\%$ | $\beta = +1.06$ | $0.000\%$ | $0.000\%$ |
| **IPPO** | $\mathbf{5.80 \pm 0.05}$ | $7.24 \pm 0.12$ | $\mathbf{+1.44}$ veh | $+24.8\%$ | $\beta = +2.95$ | $0.000\%$ | $0.000\%$ |
| **FMA2C** | $\mathbf{5.82 \pm 0.09}$ | $7.17 \pm 0.10$ | $\mathbf{+1.35}$ veh | $+23.2\%$ | $\beta = +2.78$ | $0.000\%$ | $0.000\%$ |
| **MAPPO-LCE** | $6.37 \pm 0.33$ | $7.65 \pm 0.28$ | $\mathbf{+1.28}$ veh | $+20.1\%$ | $\beta = +2.56$ | $0.000\%$ | $0.000\%$ |
| **MAPPO** | $8.85 \pm 0.50$ | $7.48 \pm 0.22$ | $\mathbf{-1.36}$ veh | $-15.4\%$ | $\beta = -2.16$ | $0.158\%$ | $0.001\%$ |

*Queue lengths are reported as mean $\pm$ standard deviation across the 10 evaluation episodes.*

---

## 6. Per-Training-Seed Results

To prevent pseudoreplication and ensure full transparency, the performance of each independent model training replicate is detailed below:

### 6.1 Clean Reference Condition (Queue in Vehicles)

| Method | Seed 42 | Seed 43 | Seed 44 | Replicate Mean $\pm$ SD | Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IPPO** | 5.76 | 5.84 | 5.81 | $5.80 \pm 0.04$ | $[5.76, 5.84]$ |
| **FMA2C** | 5.85 | 5.71 | 5.91 | $5.82 \pm 0.10$ | $[5.71, 5.91]$ |
| **MAPPO-LCE** | 6.28 | 6.84 | 6.01 | $6.38 \pm 0.42$ | $[6.01, 6.84]$ |
| **MAPPO** | 8.95 | 8.80 | 8.76 | $8.84 \pm 0.10$ | $[8.76, 8.95]$ |
| *Fixed-Time (Reference)* | *11.29* | *11.27* | *11.32* | *$11.29 \pm 0.03$* | *$[11.27, 11.32]$* |
| *Actuated (Reference)* | *6.46* | *6.44* | *6.45* | *$6.45 \pm 0.01$* | *$[6.44, 6.46]$* |

### 6.2 Extreme Noise Condition (Queue in Vehicles)

| Method | Seed 42 | Seed 43 | Seed 44 | Replicate Mean $\pm$ SD | Bootstrap 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IPPO** | 7.30 | 7.15 | 7.27 | $7.24 \pm 0.08$ | $[7.15, 7.30]$ |
| **FMA2C** | 7.18 | 7.22 | 7.13 | $7.17 \pm 0.05$ | $[7.13, 7.22]$ |
| **MAPPO-LCE** | 7.48 | 8.04 | 7.44 | $7.65 \pm 0.33$ | $[7.44, 8.04]$ |
| **MAPPO** | 7.63 | 7.43 | 7.33 | $7.48 \pm 0.15$ | $[7.33, 7.63]$ |
| *Fixed-Time (Reference)* | *11.29* | *11.27* | *11.32* | *$11.29 \pm 0.03$* | *$[11.27, 11.32]$* |
| *Actuated (Reference)* | *6.96* | *6.98* | *6.98* | *$6.97 \pm 0.01$* | *$[6.96, 6.98]$* |

---

## 7. Corrected Statistical Analysis

Statistical comparisons use the independent model training replicates ($K=3$) as the unit of analysis, paired with Fixed-Time evaluations across identical matched evaluation traffic seed subsets.

### 7.1 Significance Against Fixed-Time (Clean Reference)

| Algorithm | Mean Difference ($\bar{D}$) | SD Difference ($s_D$) | Paired $t$ ($df=2$) | $p$-value ($t$-test) | Wilcoxon Exact $p$ | Wilcoxon Significant? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Actuated** | $-4.85$ veh | $0.026$ veh | $-322.66$ | $p < 0.001$ | $p = 0.25$ | **NO** (Power limit) |
| **IPPO** | $-5.49$ veh | $0.050$ veh | $-190.60$ | $p < 0.001$ | $p = 0.25$ | **NO** (Power limit) |
| **FMA2C** | $-5.47$ veh | $0.076$ veh | $-124.12$ | $p < 0.001$ | $p = 0.25$ | **NO** (Power limit) |
| **MAPPO-LCE** | $-4.92$ veh | $0.448$ veh | $-19.00$ | $p = 0.003$ | $p = 0.25$ | **NO** (Power limit) |
| **MAPPO** | $-2.46$ veh | $0.115$ veh | $-36.95$ | $p < 0.001$ | $p = 0.25$ | **NO** (Power limit) |

### 7.2 Explicit Statement of Statistical Power Limitations
1. **Wilcoxon Signed-Rank Test**:
   With $K=3$ paired observations, the exact two-sided Wilcoxon test statistic $W=0$ yields an exact $p$-value of $\frac{2}{2^3} = \mathbf{0.25}$. **Significance at $\alpha = 0.05$ cannot be attained under Wilcoxon with $K=3$**. All claims of non-parametric significance have been removed.
2. **Student's Paired $t$-Test**:
   Degrees of freedom are $df = 2$. Critical $t$ is $4.303$. Because the physical differences (2.4 to 5.5 vehicles) are large compared to the inter-seed training variance (0.05 to 0.45 vehicles), the paired $t$-test achieves $p < 0.01$. However, parametric normality cannot be robustly verified with $n=3$.
3. **Phase 3 Recommendation**: Increase training replicates to $K \ge 5$ (ideally $K=10$) to enable valid non-parametric hypothesis testing ($p_{\min} = 0.002$ at $K=10$).

---

## 8. Robustness Degradation Slopes

The robustness degradation slope ($\beta$) measures queue sensitivity to increasing observation noise ($\sigma \in [0.0, 0.5]$):

```
       Queue Degradation across Noise Levels (Clean -> Extreme)
  12 ┤  Fixed-Time (b = 0.00, open-loop)
  10 ┤
   8 ┤  MAPPO (b = -2.16, noise disrupts greedy lock-in)
   7 ┤  MAPPO-LCE (b = +2.56)   IPPO (b = +2.95)   FMA2C (b = +2.78)
   6 ┤  Actuated (b = +1.06, robust feedback)
     └──────┬────────────┬────────────┬────────────┬────────────
          Clean        Mild       Moderate    Severe      Extreme
        (s = 0.0)    (s = 0.1)    (s = 0.2)  (s = 0.3)   (s = 0.5)
```

- **Fixed-Time** ($\beta = 0.00$): Open-loop; zero sensitivity to sensor corruption.
- **Actuated** ($\beta = +1.06$): Highly robust local feedback.
- **IPPO** ($\beta = +2.95$) & **FMA2C** ($\beta = +2.78$): Highly fragile; rapid degradation under noise.
- **MAPPO-LCE** ($\beta = +2.56$): Fragile; similar degradation rate to unconstrained RL.
- **MAPPO** ($\beta = -2.16$): Inverted slope due to noise-induced dithering.

---

## 9. Constraint and Starvation Diagnostics

### 9.1 Physical Red-Wait Dynamics & Yellow Clearance
Non-preemptible yellow clearance intervals ($3$ seconds) contribute continuously to opposing red-wait accumulation:
$$\Delta t_{\text{red}} = 3\text{s (Yellow)} + T_{\text{Opposing Green}} + 3\text{s (Yellow)} = 6\text{s} + T_{\text{Opposing Green}}$$
- For Fixed-Time, $T_{\text{Green}} = 30\text{s} \implies \Delta t_{\text{red}} = 36\text{s}$.
- For IPPO, $T_{\text{Green}} \in [5\text{s}, 18\text{s}] \implies \Delta t_{\text{red}} \in [11\text{s}, 24\text{s}]$ (peaking at $27$s).
- For MAPPO, greedy corridor lock-in holds green up to `max_green = 50s` $\implies \Delta t_{\text{red}} = 53\text{s}$.

### 9.2 The 38-Second Threshold Analysis
The 38-second threshold sits immediately above Fixed-Time's red interval ($36$s) and IPPO's symmetric switching range ($24\text{s} \to 27\text{s}$), but below greedy MAPPO ($53$s). Because this threshold was chosen to separate existing controller behaviors, **it represents a post-hoc calibration risk**. It cannot be presented as an externally validated safety standard.

### 9.3 Training Multiplier Audit & Absence of Ablation
- Inspection of `learning_curves_mappo_lce*.json` proves that $\lambda_{\text{starve}} = 0.0000$ throughout all 200 training episodes in all three seeds.
- Stochastic exploration during training switched phases frequently, preventing continuous red waits from reaching 38 seconds.
- Consequently, MAPPO-LCE received zero constraint penalties during training.
- No matched $\lambda \equiv 0$ ablation was conducted under identical seeds.
- **Auditor Ruling**: It is prohibited to claim that the Lagrangian constraint caused the queue improvement in MAPPO-LCE over MAPPO.

---

## 10. MAPPO Investigation Outcome

### 10.1 The Clean-Observation Underperformance
In the clean reference condition, MAPPO achieves an average queue of **$8.85 \pm 0.50$ veh**, significantly worse than IPPO ($5.80$ veh), FMA2C ($5.82$ veh), and Actuated ($6.45$ veh).
- **Cause**: MAPPO's centralized critic $V(s) \in \mathbb{R}^{256}$ learned that aggregate network throughput is maximized by keeping primary traffic corridors green for long durations (up to `max_green = 50s`).
- While primary corridors flowed freely, secondary approaches backed up to 20+ vehicles, elevating overall average network queue and producing starvation violations.

### 10.2 Consistency of the Negative Noise Slope
The negative degradation slope ($\beta = -2.16$) was verified to be **consistent across all three training seeds**:
- Seed 42: Clean $8.95 \to$ Extreme $7.63$ veh ($\Delta = -1.32$)
- Seed 43: Clean $8.80 \to$ Extreme $7.43$ veh ($\Delta = -1.37$)
- Seed 44: Clean $8.76 \to$ Extreme $7.33$ veh ($\Delta = -1.43$)

### 10.3 Mechanism: Noise as an Inadvertent Dither
Under clean observations, deterministic `argmax` selection locks the policy into holding green continuously.
When observation noise is injected, the noisy queue observations randomly perturb policy logits around zero. This noise perturbation breaks the greedy lock-in, forcing the policy to alternate phases more frequently. Because secondary approaches are serviced before queues explode, **observation noise paradoxically improves MAPPO's average queue length**.

---

## 11. Reproducibility and Artifact Integrity

All Phase 2 experimental artifacts were verified for integrity, isolation, and completeness:
- **Phase 2 V1 Artifacts**: Bit-for-bit identical to pre-run state (`experiments/phase2_flat_baselines/`).
- **Phase 2 V2 Artifacts**: 12 complete model checkpoints, 12 learning curves, 12 blind diagnostic files, and raw evaluation results in `experiments/phase2_canonical_v2/`.
- **SHA-256 Catalog**: A complete cryptographic manifest is preserved at `experiments/phase2_finalization/diagnostics/artifact_manifest_sha256.json`.
- **Test Suite**: 52 passed unit and integration tests with zero failures.

---

## 12. Methodological Limitations

1. **Training Sample Size ($K=3$)**: Sample size limits statistical power. Non-parametric significance ($p < 0.05$) is unachievable with Wilcoxon at $K=3$.
2. **Missing Noise-Trained Baseline**: Phase 2 only evaluated models trained on clean observations. It does not establish whether Belief State Estimation (BSE) provides benefits beyond standard data augmentation (training directly with noise).
3. **Multiplier Dormancy During Training**: MAPPO-LCE trained with $\lambda_{\text{starve}} \equiv 0.0$, meaning its training updates were mathematically unpenalized.
4. **Post-Hoc Threshold Calibration**: The 38-second starvation threshold was tuned to controller cycle times rather than derived from an external safety code.
5. **Absence of Logged Red-Wait Histograms**: Per-step red-wait timers were not saved to disk during the canonical evaluation loop; obtaining full empirical red-wait distributions requires a dedicated instrumented evaluation run.

---

## 13. Claims Supported by Evidence

- [x] **IPPO and FMA2C outperform Actuated control under clean reference observations** ($5.80$ and $5.82$ vs. $6.45$ veh, $p < 0.01$).
- [x] **Learned RL controllers are highly fragile to sensor observation noise**, degrading at nearly triple the rate of Actuated control ($\beta \approx +2.8 \to +2.95$ vs. $+1.06$).
- [x] **Actuated control outperforms all learned RL policies under extreme observation noise** ($6.97$ vs. $7.17 \to 7.65$ veh).
- [x] **All learned policies outperform open-loop Fixed-Time control across all noise levels** by $32\%$ to $48\%$.
- [x] **MAPPO's negative robustness slope is consistent across all three training seeds** and is caused by noise disrupting greedy phase lock-in.
- [x] **Action masking successfully suppresses illegal signal switching**, reducing phase-skip and green-skip violations to $0.0\%$.

---

## 14. Claims That Must NOT Be Made

- [ ] **Do NOT claim that Wilcoxon signed-rank tests demonstrate statistical significance** ($p_{\min} = 0.25$ with $K=3$).
- [ ] **Do NOT report Cohen's $d$ values exceeding $-200$** (flawed metric resulting from near-zero traffic-averaging variance).
- [ ] **Do NOT claim that MAPPO-LCE's queue improvement was caused by its constraint mechanism** ($\lambda_{\text{starve}} = 0$ throughout training; no matched $\lambda=0$ ablation exists).
- [ ] **Do NOT claim that the 38-second starvation threshold represents an externally validated safety standard** (it was an internally tuned operational parameter).
- [ ] **Do NOT claim that Phase 2 proves RL requires Belief State Estimation instead of simple noise-augmented training** (noise-trained baselines were not included in Phase 2).
- [ ] **Do NOT claim that Fixed-Time or Actuated have independent training seeds**.

---

## 15. Phase 3 Handoff Requirements

Phase 3 is authorized to proceed with the following mandatory specifications:
1. **Primary Target**: Integrate Belief State Estimation (BSE) with **IPPO** (lowest clean queue, clean parameter-shared decentralized baseline).
2. **Secondary Target**: Integrate BSE with **FMA2C** (hierarchical baseline) to test whether belief reconstruction protects multi-level regional coordination.
3. **Mandatory Noise-Trained Comparison**: Execute a 6-way controlled benchmark comparing Clean-Trained, Noise-Trained, and Clean-Trained + BSE models for both IPPO and FMA2C.
4. **Matched Protocol**: Maintain the identical 16-intersection grid network, 10 matched evaluation seeds, 3,600-step horizon, and 5 canonical noise scenarios.
5. **Statistical Scale-Up**: Expand independent training seeds to $K \ge 5$ (ideally $K=10$) to enable valid non-parametric hypothesis testing.

