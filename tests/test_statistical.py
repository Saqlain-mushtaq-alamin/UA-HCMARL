"""
Unit tests for src/evaluation/statistical.py
"""

import numpy as np
import pytest
from src.evaluation.statistical import (
    compute_confidence_interval,
    wilcoxon_test,
    compute_cohens_d,
    generate_statistical_report,
)


def test_compute_confidence_interval():
    data = [10.0, 12.0, 11.0, 10.5, 11.5]
    mean_val, low, high = compute_confidence_interval(data, confidence=0.95)
    
    assert 10.0 <= mean_val <= 12.0
    assert low <= mean_val <= high


def test_wilcoxon_test():
    sample_a = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
    sample_b = [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    
    res = wilcoxon_test(sample_a, sample_b)
    assert "p_value" in res
    assert res["p_value"] < 0.05
    assert res["significant"] == 1.0


test_cohens_d = lambda: assert_cohens_d()

def assert_cohens_d():
    a = [10.0, 11.0, 12.0]
    b = [5.0, 6.0, 7.0]
    d = compute_cohens_d(a, b)
    assert d > 1.0


def test_statistical_report():
    methods = {
        "fixed_time": [20.0, 21.0, 22.0, 19.5, 20.5],
        "actuated": [15.0, 14.5, 16.0, 15.2, 14.8],
    }
    report = generate_statistical_report(methods, baseline_name="fixed_time")
    assert "fixed_time" in report
    assert "actuated" in report
    assert "vs_baseline" in report["actuated"]
