"""
Linear Programming (LP) Projection Layer (Phase 1 / Layer 3)

Projects raw agent budget allocations onto the constrained simplex:
    sum(B_i) = B_total
    B_min <= B_i <= B_max
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class LPBudgetProjection:
    """
    Projects arbitrary continuously-valued raw budget proposals from Manager agent
    onto feasible physical constraint bounds:
        1. Exact sum preservation: sum(B_i) = B_total
        2. Box constraints: B_min <= B_i <= B_max for all i in [1..k]
    """

    def __init__(
        self,
        num_workers: int,
        b_total: float = 100.0,
        b_min: float = 5.0,
        b_max: float = 50.0,
    ):
        self.num_workers = num_workers
        self.b_total = b_total
        self.b_min = b_min
        self.b_max = b_max

        # Ensure feasible parameters
        if num_workers * b_min > b_total or num_workers * b_max < b_total:
            # Adjust bounds gracefully if infeasible
            self.b_min = max(0.0, b_total / (2 * num_workers))
            self.b_max = min(b_total, (2 * b_total) / num_workers)

    def project(self, raw_budgets: np.ndarray) -> np.ndarray:
        """
        Projects input raw budget vector raw_budgets of shape (num_workers,)
        to closest feasible point under L2 norm.
        """
        raw = np.asarray(raw_budgets, dtype=np.float64)
        if raw.ndim != 1 or len(raw) != self.num_workers:
            raise ValueError(f"Expected raw_budgets of shape ({self.num_workers},), got {raw.shape}")

        if CVXPY_AVAILABLE:
            try:
                b = cp.Variable(self.num_workers)
                objective = cp.Minimize(cp.sum_squares(b - raw))
                constraints = [
                    cp.sum(b) == self.b_total,
                    b >= self.b_min,
                    b <= self.b_max,
                ]
                prob = cp.Problem(objective, constraints)
                prob.solve(solver=cp.CLARABEL if hasattr(cp, 'CLARABEL') else None, verbose=False)

                if b.value is not None:
                    return np.array(b.value, dtype=np.float32)
            except Exception:
                pass  # Fall back to analytical solver below

        # High-performance analytical bisection / clipping projection fallback
        return self._analytical_projection(raw)

    def _analytical_projection(self, raw: np.ndarray) -> np.ndarray:
        """Analytical water-filling / clipping projection onto simplex box constraints."""
        b = np.copy(raw)
        
        # Bisection to find optimal dual parameter lambda such that sum(clip(raw - lambda)) = B_total
        low = np.min(raw) - self.b_max
        high = np.max(raw) - self.b_min

        for _ in range(50):
            mid = (low + high) / 2.0
            candidate = np.clip(raw - mid, self.b_min, self.b_max)
            cur_sum = np.sum(candidate)
            if abs(cur_sum - self.b_total) < 1e-5:
                break
            if cur_sum > self.b_total:
                low = mid
            else:
                high = mid

        res = np.clip(raw - mid, self.b_min, self.b_max)
        
        # Exact residual redistribution to guarantee exact sum equal to b_total
        residual = self.b_total - np.sum(res)
        if abs(residual) > 1e-6:
            unconstrained_indices = np.where((res > self.b_min) & (res < self.b_max))[0]
            if len(unconstrained_indices) > 0:
                res[unconstrained_indices] += residual / len(unconstrained_indices)
            else:
                res += residual / len(res)

        return res.astype(np.float32)
