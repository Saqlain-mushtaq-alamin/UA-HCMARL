"""
Layer 3 — Uncertainty-Aware Manager Agent (Phase 4 / Blueprint §6)

State: [mean_belief, mean_sigma, max_sigma, var_sigma, mean_violations, budget_remaining]
Output: Continuous budget distribution {B_1..B_k} via LP Projection Layer
Acts every T_manager steps (e.g. T_manager = 15)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from src.models.lp_projection import PyTorchLPProjectionLayer


def compute_manager_state(
    worker_reports: List[Dict[str, Any]],
    total_budget: float = 100.0,
    remaining_budget: float = 100.0,
    obs_dim: int = 16,
) -> np.ndarray:
    """
    Aggregates worker-level reports into manager state vector:
    - mean_belief: regional traffic state vector (mean across workers)
    - mean_sigma: mean uncertainty score
    - max_sigma: max uncertainty score (worst sensor)
    - var_sigma: variance of uncertainty across workers
    - mean_violations: regional constraint violation rate
    - budget_remaining: remaining budget ratio
    """
    if len(worker_reports) == 0:
        return np.zeros(obs_dim + 5, dtype=np.float32)

    beliefs = np.array([w.get("belief", np.zeros(obs_dim)) for w in worker_reports], dtype=np.float32)
    sigmas = np.array([w.get("sigma", 0.1) for w in worker_reports], dtype=np.float32)
    violations = np.array([w.get("violation_rate", 0.0) for w in worker_reports], dtype=np.float32)

    mean_b = np.mean(beliefs, axis=0)
    mean_sig = float(np.mean(sigmas))
    max_sig = float(np.max(sigmas))
    var_sig = float(np.var(sigmas))
    mean_viol = float(np.mean(violations))
    b_rem_ratio = float(remaining_budget / max(1.0, total_budget))

    state_vec = np.concatenate([mean_b, [mean_sig, max_sig, var_sig, mean_viol, b_rem_ratio]], axis=0)
    return state_vec.astype(np.float32)


class UncertaintyAwareManager(nn.Module):
    """
    PPO Actor-Critic Uncertainty-Aware Manager Agent (Layer 3).
    Generates budget allocation proposals constrained via LP Projection Layer.
    """

    def __init__(
        self,
        num_workers: int = 4,
        state_dim: int = 21,
        hidden_dim: int = 64,
        b_total: float = 100.0,
        b_min: float = 5.0,
        b_max: float = 50.0,
        smoothing_tau: float = 0.8,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.state_dim = state_dim
        self.b_total = b_total
        self.smoothing_tau = smoothing_tau

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (mean raw budgets)
        self.actor_mean = nn.Linear(hidden_dim, num_workers)
        self.actor_log_std = nn.Parameter(torch.full((num_workers,), -0.5))

        # Value head V_M
        self.value_head = nn.Linear(hidden_dim, 1)

        # PyTorch LP Projection layer (Straight-Through Estimator)
        self.lp_projector = PyTorchLPProjectionLayer(
            num_workers=num_workers, b_total=b_total, b_min=b_min, b_max=b_max
        )

        self.prev_budget: Optional[torch.Tensor] = None

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        mean_logits = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)

        dist = Normal(mean_logits, std)
        raw_budget = torch.softmax(mean_logits, dim=-1) * self.b_total

        # Pass through LP Projection Layer
        projected_budget = self.lp_projector(raw_budget)

        # Smooth budget changes to prevent oscillation (tau = 0.8)
        if self.prev_budget is not None and self.prev_budget.shape == projected_budget.shape:
            smooth_budget = self.smoothing_tau * self.prev_budget + (1.0 - self.smoothing_tau) * projected_budget
        else:
            smooth_budget = projected_budget

        self.prev_budget = smooth_budget.detach()
        value = self.value_head(features).squeeze(-1)

        return dist, smooth_budget, value
