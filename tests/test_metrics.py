"""
Unit tests for src/evaluation/metrics.py
"""

import numpy as np
import pytest
from src.evaluation.metrics import (
    MetricsCalculator,
    compute_robustness_degradation_slope,
    compute_recovery_time,
)


def test_metrics_calculator():
    calc = MetricsCalculator(num_intersections=2)
    calc.record_step(
        intersection_delays={"intersection_0": 10.0, "intersection_1": 20.0},
        intersection_queues={"intersection_0": 2.0, "intersection_1": 4.0},
        step_throughput=5,
        step_violations={"green_time": 1, "green_skip": 0, "phase_skip": 0},
        step_violation_magnitude=2.0,
    )
    
    summary = calc.compute_summary()
    assert summary["avg_delay"] == 15.0
    assert summary["avg_queue_length"] == 3.0
    assert summary["total_throughput"] == 5.0
    assert summary["constraint_violation_rate"] == 1.0
    assert summary["jains_fairness_index"] > 0.0


def test_robustness_degradation_slope():
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    delays = [10.0, 12.0, 14.0, 16.0, 20.0]
    
    slope, intercept, r_val = compute_robustness_degradation_slope(noise_levels, delays)
    assert slope > 0.0
    assert r_val > 0.95


def test_recovery_time():
    delays = [10.0, 10.0, 50.0, 30.0, 15.0, 10.0, 10.0]
    rec_time = compute_recovery_time(delays, onset_step=2, baseline_delay=10.0, tolerance=0.1)
    assert rec_time == 3  # step 2 to step 5 (5 - 2 = 3)
