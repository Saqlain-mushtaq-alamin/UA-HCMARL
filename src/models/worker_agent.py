"""
Layer 2 — Constrained PPO Worker Agent (Phase 1 / Blueprint §5)

State: [b_hat_i, sigma_hat_i, phase_state_i, allocated_budget_i]
Action: Phase selection (discrete) with action masking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


class WorkerActorCritic(nn.Module):
    """
    PPO Actor-Critic network for Layer 2 Constrained Worker Agent.
    """

    def __init__(
        self,
        state_dim: int = 24,
        action_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (policy pi_W)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Reward Critic head V_r
        self.reward_critic = nn.Linear(hidden_dim, 1)

        # Cost Critic head V_c
        self.cost_critic = nn.Linear(hidden_dim, 1)

    def forward(
        self, state: torch.Tensor, action_mask: torch.Tensor = None
    ) -> Tuple[Categorical, torch.Tensor, torch.Tensor]:
        features = self.feature_net(state)
        logits = self.actor(features)

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, -1e9)

        dist = Categorical(logits=logits)
        v_r = self.reward_critic(features)
        v_c = self.cost_critic(features)

        return dist, v_r, v_c
