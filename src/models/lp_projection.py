"""
Linear Programming (LP) Projection Layer (Phase 1 / Phase 4 / Layer 3)

Projects raw manager budget allocations onto the constrained simplex:
    sum(B_i) = B_total
    B_min <= B_i <= B_max
Features:
- Differentiable / Straight-Through Estimator PyTorch wrapper.
- High-performance analytical bisection / clipping projection solver.
"""

from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False


class LPBudgetProjection:
    """
    Projects raw budget proposals from Manager agent onto feasible physical bounds:
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
            self.b_min = max(0.0, b_total / (2 * num_workers))
            self.b_max = min(b_total, (2 * b_total) / num_workers)

    def project(self, raw_budgets: np.ndarray) -> np.ndarray:
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

        return self._analytical_projection(raw)

    def _analytical_projection(self, raw: np.ndarray) -> np.ndarray:
        b = np.copy(raw)
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
        residual = self.b_total - np.sum(res)
        if abs(residual) > 1e-6:
            unconstrained_indices = np.where((res > self.b_min) & (res < self.b_max))[0]
            if len(unconstrained_indices) > 0:
                res[unconstrained_indices] += residual / len(unconstrained_indices)
            else:
                res += residual / len(res)

        return res.astype(np.float32)


class PyTorchLPProjectionLayer(nn.Module):
    """
    PyTorch wrapper for LPBudgetProjection using straight-through gradient estimation:
        B_projected = B_raw + (B_constrained - B_raw).detach()
    """

    def __init__(
        self,
        num_workers: int,
        b_total: float = 100.0,
        b_min: float = 5.0,
        b_max: float = 50.0,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.b_total = b_total
        self.projector = LPBudgetProjection(
            num_workers=num_workers, b_total=b_total, b_min=b_min, b_max=b_max
        )

    def forward(self, raw_budgets: torch.Tensor) -> torch.Tensor:
        if raw_budgets.dim() == 1:
            raw_np = raw_budgets.detach().cpu().numpy()
            projected_np = self.projector.project(raw_np)
            projected = torch.tensor(projected_np, dtype=torch.float32, device=raw_budgets.device)
            return raw_budgets + (projected - raw_budgets).detach()

        projected_batch = []
        for b in raw_budgets:
            b_np = b.detach().cpu().numpy()
            p_np = self.projector.project(b_np)
            projected_batch.append(p_np)

        projected = torch.tensor(
            np.array(projected_batch), dtype=torch.float32, device=raw_budgets.device
        )
        return raw_budgets + (projected - raw_budgets).detach()
