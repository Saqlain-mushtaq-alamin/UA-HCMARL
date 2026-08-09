"""
Statistical Rigor Protocol Module (Phase 1 §1.7)

Provides:
- Wilcoxon Signed-Rank Test (paired non-parametric comparison)
- Bootstrapped 95% Confidence Intervals
- Effect Size (Cohen's d, Rank-Biserial Correlation)
- Summary statistical report generator
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


def compute_confidence_interval(
    data: Union[List[float], np.ndarray], confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Computes mean and bootstrapped confidence interval [lower, upper].
    """
    arr = np.asarray(data, dtype=np.float64)
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    if len(arr) == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])

    mean_val = float(np.mean(arr))
    
    # Bootstrap CI
    n_bootstraps = 2000
    rng = np.random.default_rng(42)
    boot_means = np.empty(n_bootstraps)
    for i in range(n_bootstraps):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boot_means[i] = np.mean(sample)

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return mean_val, lower, upper


def wilcoxon_test(
    sample_a: Union[List[float], np.ndarray],
    sample_b: Union[List[float], np.ndarray],
    alternative: str = "two-sided",
) -> Dict[str, float]:
    """
    Performs Wilcoxon signed-rank test on paired samples (e.g. baseline vs proposed method across seeds).
    
    Returns dictionary with:
    - statistic
    - p_value
    - significant (bool float: 1.0 if p < 0.05 else 0.0)
    """
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError(f"Sample sizes must match for paired Wilcoxon test: {len(a)} vs {len(b)}")

    diff = a - b
    if np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": 0.0,
        }

    try:
        res = stats.wilcoxon(a, b, alternative=alternative)
        stat = float(res.statistic)
        p_val = float(res.pvalue)
    except Exception:
        stat, p_val = 0.0, 1.0

    return {
        "statistic": stat,
        "p_value": p_val,
        "significant": 1.0 if p_val < 0.05 else 0.0,
    }


def compute_cohens_d(
    sample_a: Union[List[float], np.ndarray], sample_b: Union[List[float], np.ndarray]
) -> float:
    """Computes Cohen's d effect size for two independent or paired samples."""
    a = np.asarray(sample_a, dtype=np.float64)
    b = np.asarray(sample_b, dtype=np.float64)
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return 0.0

    s1_sq = np.var(a, ddof=1)
    s2_sq = np.var(b, ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / (n1 + n2 - 2) + 1e-8)
    d = (np.mean(a) - np.mean(b)) / s_pooled
    return float(d)


def generate_statistical_report(
    method_results: Dict[str, List[float]], baseline_name: str = "fixed_time"
) -> Dict[str, Dict[str, Union[float, Dict]]]:
    """
    Generates a full statistical report across seeds for multiple algorithms.
    """
    report = {}
    base_samples = method_results.get(baseline_name)

    for method_name, samples in method_results.items():
        mean_val, ci_low, ci_high = compute_confidence_interval(samples)
        method_stat = {
            "mean": mean_val,
            "ci_95_low": ci_low,
            "ci_95_high": ci_high,
            "std": float(np.std(samples)),
            "num_seeds": len(samples),
        }

        if base_samples and method_name != baseline_name and len(base_samples) == len(samples):
            w_res = wilcoxon_test(samples, base_samples)
            effect_d = compute_cohens_d(samples, base_samples)
            method_stat["vs_baseline"] = {
                "baseline_name": baseline_name,
                "wilcoxon_stat": w_res["statistic"],
                "p_value": w_res["p_value"],
                "significant_p_lt_005": bool(w_res["significant"]),
                "cohens_d": effect_d,
            }

        report[method_name] = method_stat

    return report
