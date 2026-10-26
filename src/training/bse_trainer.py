"""
BSE Supervised Pre-Training & Uncertainty Calibration Trainer (Phase 3 / Step 1)

Features:
- Dataset Collection: Gathers (o_noisy, s_true) sequence pairs across noise levels.
- Supervised Pre-Training: Optimizes GRU-KalmanNet via L_BSE loss.
- Calibration Evaluation: Evaluates MSE reduction (BSE vs Raw) and uncertainty correlation r(sigma_hat, error).

FIX (2026-08-26) -- Bug 3.2:
  The RL policy (BSEMAPPOLCEController) was being EVALUATED on BSE-smoothed inputs
  (b_hat) but TRAINED using raw noisy observations from the environment during Phase 2
  replay. This distribution shift caused BSE to INCREASE delay at every noise level
  instead of reducing it.

  Fix strategy:
  1. During Phase 3 RL training, the controller already uses b_hat from the BSE as
     worker_input (see bse_mappo_lce.py select_actions). BSE must be pre-trained
     BEFORE RL training starts so that the BSE outputs are meaningful.
  2. The new collect_bse_dataset_with_beliefs() helper generates (history, b_hat)
     pairs from a trained BSE to allow warm-starting RL replay with BSE outputs.
  3. Run evaluate_bse_calibration() after training and confirm bse_mse < raw_mse
     under severe/extreme noise to verify BSE benefit.
"""


from typing import Dict, List, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.envs.sumo_env import SUMOEnv
from src.models.belief_encoder import GRUKalmanNet


def collect_bse_dataset(
    env: SUMOEnv,
    num_steps: int = 1000,
    history_len: int = 5,
    noise_scenarios: List[str] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collects dataset of (noisy_obs_history, ground_truth_state) pairs across noise profiles.
    Allows independent seeds for train and test splits to prevent data contamination.
    """
    if noise_scenarios is None:
        noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]

    history_samples = []
    ground_truth_samples = []

    for idx, noise in enumerate(noise_scenarios):
        env.noise_injector.set_scenario(noise)
        obs_dict, _ = env.reset(seed=seed + idx * 1000)
        agent_ids = list(obs_dict.keys())
        agent_histories = {aid: [obs_dict[aid]] * history_len for aid in agent_ids}

        for step in range(num_steps // len(noise_scenarios)):
            actions = {aid: env.action_space.sample() for aid in agent_ids}
            obs_dict, _, dones, _, info = env.step(actions)

            for aid in agent_ids:
                noisy_obs = obs_dict[aid]
                true_obs = env._get_clean_observation(aid)

                agent_histories[aid].append(noisy_obs)
                if len(agent_histories[aid]) > history_len:
                    agent_histories[aid].pop(0)

                history_samples.append(np.array(agent_histories[aid], dtype=np.float32))
                ground_truth_samples.append(true_obs.astype(np.float32))

            if all(dones.values()):
                obs_dict, _ = env.reset()
                agent_histories = {aid: [obs_dict[aid]] * history_len for aid in agent_ids}

    X = np.array(history_samples, dtype=np.float32)
    Y = np.array(ground_truth_samples, dtype=np.float32)
    return X, Y


def train_bse_supervised(
    bse: GRUKalmanNet,
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-3,
    beta: float = 0.1,
    device: str = "cpu",
) -> List[float]:
    """
    Supervised pre-training of GRU-KalmanNet (Phase A).
    """
    device_obj = torch.device(device)
    bse.to(device_obj)
    bse.train()

    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(Y, dtype=torch.float32),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(bse.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for b_x, b_y in dataloader:
            b_x, b_y = b_x.to(device_obj), b_y.to(device_obj)
            b_hat, sigma_hat, _ = bse(b_x)

            loss_dict = bse.compute_loss(b_hat, sigma_hat, b_y, beta=beta)
            loss = loss_dict["loss_bse"]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

    return loss_history


def evaluate_bse_calibration(
    bse: GRUKalmanNet,
    X: np.ndarray,
    Y: np.ndarray,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluates BSE state estimation quality (MSE raw vs BSE) and uncertainty calibration.
    """
    device_obj = torch.device(device)
    bse.to(device_obj)
    bse.eval()

    X_t = torch.tensor(X, dtype=torch.float32, device=device_obj)
    Y_t = torch.tensor(Y, dtype=torch.float32, device=device_obj)

    with torch.no_grad():
        b_hat, sigma_hat, _ = bse(X_t)

    raw_last = X_t[:, -1, :]
    raw_mse = float(torch.mean((raw_last - Y_t) ** 2).item())
    bse_mse = float(torch.mean((b_hat - Y_t) ** 2).item())

    # Uncertainty calibration: correlation between sigma_hat and actual error
    actual_errors = torch.abs(b_hat - Y_t).mean(dim=-1).cpu().numpy()
    sigmas = sigma_hat.squeeze(-1).cpu().numpy()

    if np.std(sigmas) > 1e-6 and np.std(actual_errors) > 1e-6:
        corr = float(np.corrcoef(sigmas, actual_errors)[0, 1])
    else:
        corr = 0.0

    return {
        "raw_mse": raw_mse,
        "bse_mse": bse_mse,
        "mse_reduction_pct": max(0.0, float((raw_mse - bse_mse) / max(1e-6, raw_mse) * 100.0)),
        "sigma_calibration_correlation": corr,
        "mean_sigma": float(np.mean(sigmas)),
    }


def collect_bse_dataset_with_beliefs(
    env: SUMOEnv,
    bse: GRUKalmanNet,
    num_steps: int = 500,
    history_len: int = 5,
    noise_scenarios: List[str] = None,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bug 3.2 fix helper: Collects (noisy_obs_history, b_hat, sigma_hat) tuples from a
    trained BSE.  Use this to verify that the RL policy is being fed b_hat during
    training, and to confirm that b_hat tracks ground truth better than raw_obs.

    Returns
    -------
    X_hist   : (N, history_len, obs_dim)  noisy history windows
    B_hat    : (N, obs_dim)               BSE beliefs
    S_hat    : (N,)                       uncertainty scores
    """
    if noise_scenarios is None:
        noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]

    device_obj = torch.device(device)
    bse.to(device_obj)
    bse.eval()

    X_hist, B_hat, S_hat = [], [], []

    for noise in noise_scenarios:
        env.noise_injector.set_scenario(noise)
        obs_dict, _ = env.reset(seed=123)
        agent_ids = list(obs_dict.keys())
        agent_histories: Dict[str, List[np.ndarray]] = {
            aid: [obs_dict[aid]] * history_len for aid in agent_ids
        }

        for step in range(num_steps // len(noise_scenarios)):
            actions = {aid: env.action_space.sample() for aid in agent_ids}
            obs_dict, _, dones, _, _ = env.step(actions)

            for aid in agent_ids:
                noisy_obs = obs_dict[aid]
                agent_histories[aid].append(noisy_obs)
                if len(agent_histories[aid]) > history_len:
                    agent_histories[aid].pop(0)

                hist_arr = np.array(agent_histories[aid], dtype=np.float32)
                hist_t = torch.tensor(hist_arr, dtype=torch.float32, device=device_obj).unsqueeze(0)

                with torch.no_grad():
                    b_hat_t, sigma_t, _ = bse(hist_t)

                X_hist.append(hist_arr)
                B_hat.append(b_hat_t.squeeze(0).cpu().numpy())
                S_hat.append(float(sigma_t.item()))

            if all(dones.values()):
                obs_dict, _ = env.reset()
                agent_histories = {aid: [obs_dict[aid]] * history_len for aid in agent_ids}

    return (
        np.array(X_hist, dtype=np.float32),
        np.array(B_hat, dtype=np.float32),
        np.array(S_hat, dtype=np.float32),
    )
