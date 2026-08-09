"""
Layer 3 — Uncertainty-Aware Manager Agent (Phase 1 / Blueprint §6)

State: [b_bar_region, sigma_bar_region, sigma_max, V_bar_violations, B_remaining, demand]
Output: Continuous budget distribution {B_1..B_k} via LP Projection Layer
Acts every T_manager steps (e.g. T_manager = 15)
"""

from typing import Tuple
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from src.models.lp_projection import LPBudgetProjection


class UncertaintyAwareManager(nn.Module):
    """
    PPO Actor-Critic Regional Manager Agent (Layer 3).
    Output: Raw budget vector projected through LP projection layer.
    """

    def __init__(
        self,
        num_workers: int,
        state_dim: int = 16,
        hidden_dim: int = 64,
        b_total: float = 100.0,
        b_min: float = 5.0,
        b_max: float = 50.0,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.state_dim = state_dim
        self.b_total = b_total

        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (mean raw budgets)
        self.actor_mean = nn.Linear(hidden_dim, num_workers)
        self.actor_log_std = nn.Parameter(torch.zeros(num_workers))

        # Value head V_M
        self.value_head = nn.Linear(hidden_dim, 1)

        # LP projection layer
        self.lp_projector = LPBudgetProjection(
            num_workers=num_workers, b_total=b_total, b_min=b_min, b_max=b_max
        )

    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.feature_net(state)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std)

        dist = Normal(mean, std)
        value = self.value_head(features)
        return dist, value

    def project_budget(self, raw_budgets: torch.Tensor) -> torch.Tensor:
        """Projects raw output vector through LP projector."""
        if raw_budgets.dim() == 1:
            raw_np = raw_budgets.detach().cpu().numpy()
            projected = self.lp_projector.project(raw_np)
            return torch.tensor(projected, dtype=torch.float32, device=raw_budgets.device)

        projected_batch = []
        for b in raw_budgets:
            b_np = b.detach().cpu().numpy()
            p_np = self.lp_projector.project(b_np)
            projected_batch.append(p_np)

        return torch.tensor(projected_batch, dtype=torch.float32, device=raw_budgets.device)
