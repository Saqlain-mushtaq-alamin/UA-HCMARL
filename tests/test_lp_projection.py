"""
Unit tests for src/models/lp_projection.py
"""

import numpy as np
import pytest
from src.models.lp_projection import LPBudgetProjection


def test_lp_budget_projection():
    proj = LPBudgetProjection(num_workers=4, b_total=100.0, b_min=5.0, b_max=50.0)
    raw = np.array([10.0, 70.0, 0.0, 20.0], dtype=np.float32)
    
    projected = proj.project(raw)
    
    assert projected.shape == (4,)
    assert np.isclose(np.sum(projected), 100.0, atol=1e-3)
    assert np.all(projected >= 5.0 - 1e-4)
    assert np.all(projected <= 50.0 + 1e-4)
