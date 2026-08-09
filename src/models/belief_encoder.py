"""
Layer 1 — Belief-State Encoder (BSE) Module Skeleton (Phase 1 / Blueprint §4)

Architecture: GRU-KalmanNet (inspired by Q-Net)
Output: Denoised belief state b_hat + Uncertainty score sigma_hat
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUKalmanNet(nn.Module):
    """
    GRU-KalmanNet Belief-State Encoder (Layer 1).
    Encodes raw noisy observations o_i(t) over history window H into:
        - Cleaned belief state b_hat_i(t) in R^d
        - Calibrated uncertainty score sigma_hat_i(t) in R_+ (softplus)
    """

    def __init__(
        self,
        raw_obs_dim: int = 16,
        hidden_dim: int = 64,
        belief_dim: int = 16,
        history_len: int = 5,
    ):
        super().__init__()
        self.raw_obs_dim = raw_obs_dim
        self.hidden_dim = hidden_dim
        self.belief_dim = belief_dim
        self.history_len = history_len

        # GRU temporal encoder
        self.gru = nn.GRU(
            input_size=raw_obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Kalman Gain estimator network MLP_K
        self.kalman_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, raw_obs_dim * belief_dim),
        )

        # Observation projection & Belief update
        self.belief_head = nn.Linear(hidden_dim, belief_dim)

        # Uncertainty head sigma_hat (softplus output)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self, obs_history: torch.Tensor, hidden_state: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_history: (batch_size, sequence_length, raw_obs_dim)
            hidden_state: optional GRU hidden state

        Returns:
            b_hat: Denoised belief state (batch_size, belief_dim)
            sigma_hat: Uncertainty score (batch_size, 1)
            next_hidden: GRU hidden state
        """
        if obs_history.dim() == 2:
            obs_history = obs_history.unsqueeze(1)

        gru_out, next_hidden = self.gru(obs_history, hidden_state)
        last_h = gru_out[:, -1, :]  # (batch_size, hidden_dim)

        b_hat = self.belief_head(last_h)
        sigma_hat = self.uncertainty_head(last_h) + 1e-4

        return b_hat, sigma_hat, next_hidden

    def compute_loss(
        self,
        b_hat: torch.Tensor,
        sigma_hat: torch.Tensor,
        ground_truth_state: torch.Tensor,
        beta: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Auxiliary supervised loss L_BSE = MSE(b_hat, s_true) + beta * NLL(sigma_hat, |b_hat - s_true|)."""
        mse_loss = F.mse_loss(b_hat, ground_truth_state)
        err = torch.abs(b_hat - ground_truth_state).mean(dim=-1, keepdim=True)
        
        # NLL loss for uncertainty calibration
        nll_loss = (torch.log(sigma_hat) + (err ** 2) / (2 * (sigma_hat ** 2))).mean()
        
        total_loss = mse_loss + beta * nll_loss
        return {
            "loss_bse": total_loss,
            "loss_mse": mse_loss,
            "loss_nll": nll_loss,
        }
