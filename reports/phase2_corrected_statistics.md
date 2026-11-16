# Phase 2 — Corrected Statistical Methodology and Significance Report

**Date**: November 6, 2026  
**Auditor**: Antigravity Quality Assurance & Verification Agent  
**Artifact Scope**: `experiments/phase2_canonical_v2/`  
**Derived Data**: `experiments/phase2_finalization/corrected_statistical_summary.json`  

---

## 1. Executive Summary

This report rectifies statistical reporting defects identified in the preliminary Phase 2 milestone summaries:
1. **Flawed Effect Sizes Withdrawn**: Cohen's $d$ values exceeding $-200$ (e.g., Actuated $d = -228.08$, IPPO $d = -165.56$) were caused by dividing physical queue reductions ($\approx 4.85$ to $5.5$ veh) by the near-zero variance of 3,600-step traffic evaluation averages ($\sigma \approx 0.02$ veh). These misleading figures have been withdrawn and replaced with physically grounded paired differences.
2. **Wilcoxon Power Inadequacy with $K=3$**: With $K=3$ independent training replicates, the minimum attainable exact two-sided Wilcoxon signed-rank $p$-value is **$0.25$**. Reaching $p < 0.05$ under Wilcoxon is mathematically impossible at $K=3$. Preliminary reports labeling $p = 0.25$ as "significant" were invalid and are retracted.
3. **Rigorous Replicate Unit Enforcement**: Model training replicates (Seeds 42, 43, 44) are maintained as the strict unit of replication for all RL algorithms ($K=3$). Classical baselines (Fixed-Time, Actuated) are correctly reported as non-learning evaluation references without artificial training-seed replication.
4. **Bootstrap Methodology**: 95% confidence intervals are computed via percentile bootstrap directly over the $K=3$ independent replicate means with $B = 2,000$ resamples.

---

## 2. Mathematical Corrections

### 2.1 Investigation of the Inflated Cohen's $d$ Effect Sizes
In `phase2_statistical_summary.json`, Cohen's $d$ was reported as:
- Actuated vs. Fixed-Time: $d = -228.08$
- IPPO vs. Fixed-Time: $d = -165.56$
- FMA2C vs. Fixed-Time: $d = -74.10$
- MAPPO vs. Fixed-Time: $d = -33.08$

#### Root Cause Analysis:
In `src/evaluation/statistical.py` ([`compute_cohens_d`](file:///d:/canvas/MARLTSC/src/evaluation/statistical.py#L124-L138)):
$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}}$$
$$d = \frac{\bar{X}_1 - \bar{X}_2}{s_{\text{pooled}}}$$
For Fixed-Time and Actuated, the evaluation loop partitioned 10 evaluation seeds into 3 subsets ($s_{\text{idx}} \in \{0, 1, 2\}$).
Because each subset averaged $3$ to $4$ full one-hour simulation episodes (3,600 steps each) on deterministic or rule-based controllers, the law of large numbers suppressed Poisson traffic stochasticity:
$$s_{\text{Fixed-Time}} = 0.0275 \text{ veh}, \quad s_{\text{Actuated}} = 0.0119 \text{ veh} \implies s_{\text{pooled}} = 0.0212 \text{ veh}$$
Dividing the substantial physical queue reduction ($\bar{X}_{\text{Actuated}} - \bar{X}_{\text{Fixed-Time}} = 6.45 - 11.29 = -4.85$ veh) by $0.0212$ veh produced:
$$d = \frac{-4.845}{0.0212} = -228.08$$
**Scientific Correction**: Cohen's $d$ assumes independent single-observation variance, not macro-aggregated sample means. Dividing macro-differences by tiny traffic-averaging residuals produces meaningless effect sizes that confound statistical power with effect magnitude. These pooled Cohen's $d$ metrics are **retracted**. We report the raw physical mean difference $\bar{D} = \bar{X}_1 - \bar{X}_2$ (vehicles) and the paired difference effect size:
$$d_z = \frac{\bar{D}}{s_D}$$

---

### 2.2 Mathematical Proof of Wilcoxon Signed-Rank Power Limit at $K=3$
The Wilcoxon signed-rank test ranks absolute differences $|D_k| = |X_{k} - Y_{k}|$ for $k \in \{1, \dots, K\}$.
For $K = 3$ pairs:
- The possible absolute ranks are $\{1, 2, 3\}$.
- The total sum of ranks is $\frac{3(3+1)}{2} = 6$.
- Under the null hypothesis $H_0$, each rank receives a positive or negative sign with probability $0.5$.
- The number of possible sign combinations is $2^K = 2^3 = 8$.
- The minimum possible test statistic $W$ occurs when all differences have the identical sign:
  $$W = 0$$
- The exact probability of observing $W = 0$ in a two-sided test is:
  $$p_{\text{exact}} = 2 \times \left( \frac{1}{2^3} \right) = \frac{2}{8} = 0.25$$

**Conclusion**: Under an exact two-sided Wilcoxon signed-rank test with $K=3$, **$p = 0.25$ is the absolute theoretical minimum**. It is mathematically impossible to reach $p < 0.05$. Therefore, labeling $p=0.25$ as "statistically significant" was a major defect. Wilcoxon **cannot** be used to claim statistical significance at $K=3$.

---

### 2.3 Paired Student's $t$-Test with $K=3$ ($df = 2$)
The paired $t$-statistic is:
$$t = \frac{\bar{D}}{s_D / \sqrt{K}}$$
With $K = 3$, degrees of freedom $df = K - 1 = 2$.
The two-sided critical value for $\alpha = 0.05$ at $df=2$ is:
$$t_{\text{crit}} = 4.303$$
To achieve $p < 0.05$, the mean difference must exceed $4.303 \times \frac{s_D}{\sqrt{3}} \approx 2.48 \times s_D$.

---

## 3. Independent Training-Seed Results Table

For the learning algorithms (IPPO, MAPPO, MAPPO-LCE, FMA2C), each model was trained from scratch using three independent seeds:
- Seed 42 (`{method}.pt`)
- Seed 43 (`{method}_s43.pt`)
- Seed 44 (`{method}_s44.pt`)

Each trained checkpoint was evaluated across the matched traffic seeds in the clean-observation reference condition.

### 3.1 Clean Reference Condition (Queue in Vehicles)

| Algorithm | Replicate 0 (Seed 42) | Replicate 1 (Seed 43) | Replicate 2 (Seed 44) | Replicate Mean $\pm$ SD | 95% Bootstrap CI | Paired Diff vs Fixed-Time ($\bar{D}$) | Paired $t$ ($df=2$) | $p$-value ($t$-test) | $p$-value (Wilcoxon) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | 11.29* | 11.27* | 11.32* | $11.29 \pm 0.03$ | $[11.27, 11.32]$ | Baseline | — | — | — |
| **Actuated** | 6.46* | 6.44* | 6.45* | $6.45 \pm 0.01$ | $[6.44, 6.46]$ | $-4.85$ veh | $t = -322.66$ | $p < 0.001$ | $p = 0.25$ (NS) |
| **IPPO** | **5.76** | **5.84** | **5.81** | $\mathbf{5.80 \pm 0.04}$ | $[5.76, 5.84]$ | **$-5.49$ veh** | $t = -190.60$ | $p < 0.001$ | $p = 0.25$ (NS) |
| **FMA2C** | **5.85** | **5.71** | **5.91** | $\mathbf{5.82 \pm 0.10}$ | $[5.71, 5.91]$ | **$-5.47$ veh** | $t = -124.12$ | $p < 0.001$ | $p = 0.25$ (NS) |
| **MAPPO-LCE** | 6.28 | 6.84 | 6.01 | $6.38 \pm 0.42$ | $[6.01, 6.84]$ | $-4.92$ veh | $t = -19.00$ | $p = 0.003$ | $p = 0.25$ (NS) |
| **MAPPO** | 8.95 | 8.80 | 8.76 | $8.84 \pm 0.10$ | $[8.76, 8.95]$ | $-2.46$ veh | $t = -36.95$ | $p < 0.001$ | $p = 0.25$ (NS) |

*\*Note on Classical Baselines: Fixed-Time and Actuated do not have training seeds. The values shown in the columns are their evaluations on the matching traffic seed partitions (Partition 0: eval seeds 0, 3, 6, 9; Partition 1: eval seeds 1, 4, 7; Partition 2: eval seeds 2, 5, 8).*

---

### 3.2 Extreme Noise Condition (Queue in Vehicles)

| Algorithm | Replicate 0 (Seed 42) | Replicate 1 (Seed 43) | Replicate 2 (Seed 44) | Replicate Mean $\pm$ SD | Clean $\to$ Extreme Change ($\Delta$) | Degradation Slope ($\beta$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed-Time** | 11.29 | 11.27 | 11.32 | $11.29 \pm 0.03$ | $\mathbf{0.00}$ veh ($0.0\%$) | $\beta = 0.00$ |
| **Actuated** | 6.96 | 6.98 | 6.98 | $6.97 \pm 0.01$ | $\mathbf{+0.52}$ veh ($+8.1\%$) | $\beta = +1.06$ |
| **IPPO** | 7.30 | 7.15 | 7.27 | $7.24 \pm 0.08$ | $\mathbf{+1.44}$ veh ($+24.8\%$) | $\beta = +2.95$ |
| **FMA2C** | 7.18 | 7.22 | 7.13 | $7.17 \pm 0.05$ | $\mathbf{+1.35}$ veh ($+23.2\%$) | $\beta = +2.78$ |
| **MAPPO-LCE** | 7.48 | 8.04 | 7.44 | $7.65 \pm 0.33$ | $\mathbf{+1.28}$ veh ($+20.1\%$) | $\beta = +2.56$ |
| **MAPPO** | 7.63 | 7.43 | 7.33 | $7.48 \pm 0.15$ | $\mathbf{-1.36}$ veh ($-15.4\%$) | $\beta = -2.16$ |

---

## 4. Key Scientific Findings & Statistical Distinctions

1. **IPPO and FMA2C Outperform Actuated Under Clean Observations**:
   - Both IPPO ($5.80 \pm 0.04$ veh) and FMA2C ($5.82 \pm 0.10$ veh) achieve significantly lower queue lengths than Actuated ($6.45 \pm 0.01$ veh) under clean reference observations (paired difference $\approx -0.64$ veh, $p < 0.01$).
2. **Fragility Under Sensor Noise**:
   - The queue advantages of IPPO and FMA2C completely evaporate under observation noise.
   - At extreme noise, Actuated achieves **$6.97$ veh**, outperforming FMA2C ($7.17$ veh) and IPPO ($7.24$ veh).
   - RL policies exhibit nearly triple the degradation rate of Actuated ($\beta_{\text{IPPO}} = +2.95, \beta_{\text{FMA2C}} = +2.78$ vs. $\beta_{\text{Actuated}} = +1.06$).
3. **All Learned Policies Outperform Fixed-Time Across All Conditions**:
   - Even under extreme sensor corruption, all learned controllers maintain a 3.6 to 4.1 vehicle queue advantage over open-loop Fixed-Time control ($11.29$ veh).
4. **Statistical Testing Limitations**:
   - While paired $t$-tests show $p < 0.01$ against Fixed-Time due to the large magnitude of difference relative to variance, the sample size of $K=3$ limits generalization across model training runs. Phase 3 scale-up requires expanding training seeds to $K \ge 5$ (or $K \ge 10$) to enable valid non-parametric hypothesis testing.

