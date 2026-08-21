"""
Layer 1 — Belief-State Encoder (BSE) Module (Phase 3 / Blueprint §4)

Architecture: GRU-KalmanNet
Outputs:
- Denoised belief state b_hat_i(t) in R^d
- Calibrated uncertainty score sigma_hat_i(t) in R_+ (Softplus output)
Features:
- Learned Kalman gain K(t) in [0, 1]^d
- Prediction-correction update: b_hat = (1 - K) * f(b_prev) + K * o_current
- Auxiliary supervised loss: L_BSE = MSE(b_hat, s_true) + beta * NLL(sigma_hat, |b_hat - s_true|)
"""

from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUKalmanNet(nn.Module):
    """
    GRU-KalmanNet Belief-State Encoder (Layer 1).
    Encodes raw noisy observation history o_i(t-H:t) into:
        - Denoised belief state b_hat_i(t)
        - Calibrated uncertainty score sigma_hat_i(t)
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

        # GRU temporal context encoder
        self.gru = nn.GRU(
            input_size=raw_obs_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # Learned Kalman gain estimator K(t) in [0, 1]
        self.kalman_gain_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, belief_dim),
            nn.Sigmoid(),
        )

        # Transition model f(b̂(t-1))
        self.transition_net = nn.Linear(belief_dim, belief_dim)

        # Innovation-aware uncertainty head sigma_hat (Softplus)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim + belief_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        obs_history: torch.Tensor,
        prev_belief: Optional[torch.Tensor] = None,
        hidden_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs_history: Tensor of shape (batch_size, sequence_length, raw_obs_dim)
            prev_belief: Optional previous belief state (batch_size, belief_dim)
            hidden_state: Optional GRU hidden state

        Returns:
            b_hat: Denoised belief state (batch_size, belief_dim)
            sigma_hat: Uncertainty score (batch_size, 1)
            next_hidden: GRU hidden state
        """
        if obs_history.dim() == 2:
            obs_history = obs_history.unsqueeze(1)

        batch_size = obs_history.size(0)
        current_obs = obs_history[:, -1, :]

        gru_out, next_hidden = self.gru(obs_history, hidden_state)
        z_last = gru_out[:, -1, :]  # (batch_size, hidden_dim)

        # 1. Learned Kalman gain K(t)
        kalman_k = self.kalman_gain_net(z_last)

        # 2. Prediction-Correction Update
        if prev_belief is None:
            prev_belief = current_obs

        predicted_belief = self.transition_net(prev_belief)
        b_hat = (1.0 - kalman_k) * predicted_belief + kalman_k * current_obs

        # 3. Innovation-Aware Uncertainty Score
        innovation = torch.abs(current_obs - b_hat)
        uncertainty_input = torch.cat([z_last, innovation], dim=-1)
        sigma_hat = self.uncertainty_head(uncertainty_input) + 1e-4

        return b_hat, sigma_hat, next_hidden

    def compute_loss(
        self,
        b_hat: torch.Tensor,
        sigma_hat: torch.Tensor,
        ground_truth_state: torch.Tensor,
        beta: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Auxiliary loss: L_BSE = MSE(b_hat, s_true) + beta * NLL(sigma_hat, |b_hat - s_true|)
        """
        mse_loss = F.mse_loss(b_hat, ground_truth_state)
        error = torch.abs(b_hat - ground_truth_state).mean(dim=-1, keepdim=True)

        # NLL loss for uncertainty calibration
        nll_loss = (torch.log(sigma_hat) + (error ** 2) / (2.0 * (sigma_hat ** 2))).mean()

        total_loss = mse_loss + beta * nll_loss
        return {
            "loss_bse": total_loss,
            "loss_mse": mse_loss,
            "loss_nll": nll_loss,
        }
