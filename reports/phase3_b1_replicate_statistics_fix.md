# Phase 3 Remediation Report: B1 — Replicate Execution & Statistical Aggregation

## 1. Original Defect Summary
In `reports/phase3_independent_blockers.md` (Defect B1), the independent audit established that:
- `scripts/run_phase3_benchmark.py` only evaluated a single training seed (`[42]`), ignoring seeds 43 and 44 during benchmark execution.
- Evaluation episodes ($N=30$) were treated as the statistical experimental unit rather than independent training replicates ($K=3$), committing pseudo-replication.
- Claims of statistical significance (e.g. $p < 0.05$ with $N=30$) were scientifically invalid because variance across training seeds was not reflected in the standard errors or hypothesis tests.

---

## 2. Exact Source Files & Functions Changed
1. **`experiment_config.py`**:
   - Class `ExperimentConfig`: Added explicit `training_seeds: list[int] = field(default_factory=lambda: [42, 43, 44])`.
   - Function `get_config(profile="canonical")`: Configured canonical profile to provide `training_seeds=[42, 43, 44]` ($K=3$), while smoke profile sets `training_seeds=[42]` ($K=1$).
2. **`scripts/run_phase3_benchmark.py`**:
   - `run_phase3_benchmark()`: Refactored main benchmark execution loop to iterate over `cfg.training_seeds`:
     - Arm M1 & Arm M4: Clean controllers loaded per seed (`models/ippo_seed{train_seed}.pt`, `models/fma2c_seed{train_seed}.pt`).
     - Arm M2 & Arm M5: Noise-augmented controllers trained/loaded per seed (`models/ippo_noise_s{train_seed}.pt`, `models/fma2c_noise_s{train_seed}.pt`).
     - Arm M3 & Arm M6: Clean controllers + BSE wrapper instantiated per seed.
     - Per-replicate raw evaluation records saved independently to `experiments/.../replicates/raw_results_s{train_seed}.json`.
   - `compute_replicate_statistics()`: Aggregates replicate-level metric means $\bar{m}_k$ ($k=1 \dots K$), calculating:
     $$\bar{m} = \frac{1}{K}\sum_{k=1}^K \bar{m}_k, \quad s_K = \sqrt{\frac{1}{K-1}\sum_{k=1}^K (\bar{m}_k - \bar{m})^2}, \quad \text{SE} = \frac{s_K}{\sqrt{K}}$$
     $$95\% \text{ CI} = \bar{m} \pm t_{crit}(df=K-1) \cdot \text{SE}$$
   - `compute_paired_replicate_tests()`: Evaluates paired Student's $t$-tests and Wilcoxon signed-rank tests across replicate means ($K=3$). Includes explicit statistical note documenting minimum attainable Wilcoxon p-value for $K=3$ ($p_{\min} = 2 \times 0.5^3 = 0.250$).

---

## 3. Implementation Details

### Mathematical Formulation
For each method $m$, noise scenario $s$, and metric $y \in \{\text{queue\_length}, \text{delay}, \text{starvation\_rate}\}$:
1. Replicate Mean:
   $$\bar{y}_{m,s,k} = \frac{1}{E} \sum_{e=1}^E y_{m,s,k,e}$$
   where $E = N_{\text{seeds}} \times E_{\text{per\_seed}} = 10 \times 3 = 30$ evaluation episodes for replicate $k$.
2. Grand Mean & Replicate Variance ($K=3$):
   $$\bar{Y}_{m,s} = \frac{1}{3} \sum_{k=1}^3 \bar{y}_{m,s,k}$$
   $$s^2_{m,s} = \frac{1}{2} \sum_{k=1}^3 (\bar{y}_{m,s,k} - \bar{Y}_{m,s})^2$$
   $$\text{SE}(\bar{Y}_{m,s}) = \frac{s_{m,s}}{\sqrt{3}}$$
3. Hypothesis Testing:
   Pairwise differences $d_k = \bar{y}_{\text{bse},s,k} - \bar{y}_{\text{clean},s,k}$ for $k \in \{42, 43, 44\}$.
   Paired $t$-statistic on $df=2$:
   $$t = \frac{\bar{d}}{s_d / \sqrt{3}}$$

---

## 4. Tests Added & Test Verification
- **Test File**: `tests/test_statistical.py`
  - Validated that `paired_t_test` and `wilcoxon_test` calculate correct statistics and degrees of freedom.
- **Integration Test**: `tests/test_phase3_runner.py::test_six_arm_smoke_execution`
  - Validated that replicate directory structure is created and populated with per-replicate files (`raw_results_s42.json`).

### Verification Command & Output
```bash
python -m pytest tests/test_statistical.py tests/test_phase3_runner.py -v
```
Output:
```
tests/test_statistical.py::test_paired_t_test PASSED
tests/test_statistical.py::test_wilcoxon_test PASSED
tests/test_statistical.py::test_replicate_level_aggregation PASSED
tests/test_statistical.py::test_bootstrap_confidence_intervals PASSED
tests/test_phase3_runner.py::test_six_arm_smoke_execution PASSED
============================== 5 passed in 13.12s ==============================
```

---

## 5. Remaining Limitations
1. With $K=3$ independent training replicates, non-parametric Wilcoxon signed-rank tests cannot reach $\alpha = 0.05$ ($p_{\min} = 0.250$ for two-tailed test with $n=3$). Parametric Student's $t$ or bootstrap CI must serve as primary inferential measures.
2. In smoke profile, $K=1$, so $s_K$, SE, and $t$-tests default to 0.0 / NaN; multi-replicate statistics activate fully under canonical execution ($K=3$).

---

## 6. Evidence Paths
- `experiment_config.py` (lines 48–60)
- `scripts/run_phase3_benchmark.py` (lines 310–395, 430–520)
- `experiments/phase3_smoke_b1b4/replicates/raw_results_s42.json`
- `experiments/phase3_smoke_b1b4/phase3_statistical_summary.json`

