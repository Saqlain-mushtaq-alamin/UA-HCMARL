"""
PID-Lagrangian Cost Estimator Module (Phase 1 / Layer 2)

Implements Confidence-Modulated PID Lagrangian multiplier updates (Blueprint §5):
    Standard:  lambda_j <- max(0, lambda_j + alpha * (C_hat_j - d_j))
    Ours:      lambda_j <- max(0, lambda_j + alpha * (C_hat_j - d_j * gamma(sigma_hat, B)))
"""

import numpy as np


class PIDLagrangianMultiplier:
    """
    PID-Lagrangian multiplier manager with confidence modulation gamma(sigma_hat, B).
    """

    def __init__(
        self,
        num_constraints: int = 3,
        kp: float = 0.05,
        ki: float = 0.005,
        kd: float = 0.01,
        initial_lambda: float = 0.0,
        max_lambda: float = 100.0,
    ):
        self.num_constraints = num_constraints
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_lambda = max_lambda

        self.lambdas = np.full(num_constraints, initial_lambda, dtype=np.float32)
        self.integral_errors = np.zeros(num_constraints, dtype=np.float32)
        self.prev_errors = np.zeros(num_constraints, dtype=np.float32)

    def compute_gamma(self, sigma_hat: float, budget: float) -> float:
        """
        Confidence-modulation factor gamma(sigma_hat, B) in (0, 1).
        Low confidence + small budget -> gamma -> 0 -> stricter constraints.
        """
        w1, w2, bias = -2.0, 0.05, 0.0
        log_sigma = np.log(max(1e-4, sigma_hat))
        logit = w1 * log_sigma + w2 * budget + bias
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
        derivative = error - self.prev_errors
        self.prev_errors = np.copy(error)

        delta = self.kp * error + self.ki * self.integral_errors + self.kd * derivative
        self.lambdas = np.clip(self.lambdas + delta, 0.0, self.max_lambda)

        return self.lambdas
