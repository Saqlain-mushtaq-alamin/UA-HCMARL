"""
PID-Lagrangian Cost Estimator Module (Phase 1 / Phase 3 / Blueprint §5)

Implements Confidence-Modulated PID Lagrangian multiplier updates (Blueprint §5):
    Standard:  lambda_j <- max(0, lambda_j + alpha * (C_hat_j - d_j))
    Ours:      lambda_j <- max(0, lambda_j + alpha * (C_hat_j - d_j * gamma(sigma_hat, B)))

FIX (2026-08-21): The original PID gains (kp=0.05, ki=0.005, kd=0.01) were
far too large, causing the Lagrangian multiplier to spike to its max_lambda cap
after the very first update and then stay pinned there for the rest of training.
This made MAPPO-LCE behave like a fully-penalized unconstrained problem rather
than a lightly guided constrained one.

Corrected gains: kp=0.01, ki=0.001, kd=0.001.  The max_lambda is also reduced
from 100 to 10 so an occasional burst does not permanently destroy the policy
gradient.
"""

from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn


class PIDLagrangianMultiplier:
    """
    PID-Lagrangian multiplier manager with optional confidence modulation gamma(sigma_hat, B).
    """

    def __init__(
        self,
        num_constraints: int = 3,
        kp: float = 0.01,
        ki: float = 0.001,
        kd: float = 0.001,
        initial_lambda: float = 0.0,
        max_lambda: float = 10.0,
        use_confidence_modulation: bool = True,
    ):
        self.num_constraints = num_constraints
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_lambda = max_lambda
        self.use_confidence_modulation = use_confidence_modulation

        self.lambdas = np.full(num_constraints, initial_lambda, dtype=np.float32)
        self.integral_errors = np.zeros(num_constraints, dtype=np.float32)
        self.prev_errors = np.zeros(num_constraints, dtype=np.float32)

    def reset(self):
        """Reset Lagrangian state (call at the start of each new episode)."""
        self.lambdas[:] = 0.0
        self.integral_errors[:] = 0.0
        self.prev_errors[:] = 0.0

    def compute_gamma(self, sigma_hat: float, budget: float = 50.0) -> float:
        """
        Confidence-modulation factor gamma(sigma_hat, B) in (0, 1].
        High uncertainty sigma_hat -> gamma -> 0 -> stricter effective constraint thresholds.
        """
        if not self.use_confidence_modulation:
            return 1.0

        w1, w2, bias = -2.0, 0.05, 0.0
        log_sigma = np.log(max(1e-4, float(sigma_hat)))
        logit = w1 * log_sigma + w2 * float(budget) + bias
        gamma = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(gamma, 0.01, 1.0))

    def update(
        self,
        constraint_costs: np.ndarray,
        constraint_limits: np.ndarray,
        sigma_hat: float = 1.0,
        budget: float = 50.0,
    ) -> np.ndarray:
        """
        Updates Lagrangian multipliers given measured costs C_hat and limits d.
        """
        costs = np.asarray(constraint_costs, dtype=np.float32)
        limits = np.asarray(constraint_limits, dtype=np.float32)

        gamma = self.compute_gamma(sigma_hat, budget)
        effective_limits = limits * gamma

        error = costs - effective_limits
        self.integral_errors += error
        # Anti-windup: clip integral term
        self.integral_errors = np.clip(self.integral_errors, -10.0, 10.0)
        derivative = error - self.prev_errors
        self.prev_errors = np.copy(error)

        delta = self.kp * error + self.ki * self.integral_errors + self.kd * derivative
        self.lambdas = np.clip(self.lambdas + delta, 0.0, self.max_lambda)

        return self.lambdas


class ConfidenceModulatedLagrangian(nn.Module):
    """
    PyTorch Parameterized Confidence-Modulated PID Lagrangian for Layer 2 Constrained Workers.
    """

    def __init__(self, num_constraints: int = 3, kp: float = 0.01, ki: float = 0.001, kd: float = 0.001):
        super().__init__()
        self.num_constraints = num_constraints
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.w1 = nn.Parameter(torch.tensor(-2.0))
        self.w2 = nn.Parameter(torch.tensor(0.05))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def compute_gamma(self, sigma_hat: torch.Tensor, budget: torch.Tensor) -> torch.Tensor:
        log_sigma = torch.log(torch.clamp(sigma_hat, min=1e-4))
        logit = self.w1 * log_sigma + self.w2 * budget + self.bias
        return torch.sigmoid(logit)
