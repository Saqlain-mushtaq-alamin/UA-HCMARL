"""
Unit tests for Layer 1 Belief State Encoder & Stage 3 BSE-MAPPO-LCE (Phase 3)
"""

import pytest
import numpy as np
import torch
from src.models.belief_encoder import GRUKalmanNet
from src.models.lagrangian import ConfidenceModulatedLagrangian, PIDLagrangianMultiplier
from src.baselines.bse_mappo_lce import BSEMAPPOLCEController
from src.training.bse_trainer import (
    collect_bse_dataset,
    train_bse_supervised,
    evaluate_bse_calibration,
)
from src.envs.sumo_env import SUMOEnv


def test_gru_kalman_net():
    batch_size = 4
    seq_len = 5
    obs_dim = 16

    bse = GRUKalmanNet(raw_obs_dim=obs_dim, hidden_dim=32, belief_dim=obs_dim, history_len=seq_len)
    obs_history = torch.randn(batch_size, seq_len, obs_dim)

    b_hat, sigma_hat, next_hidden = bse(obs_history)

    assert b_hat.shape == (batch_size, obs_dim)
    assert sigma_hat.shape == (batch_size, 1)
    assert (sigma_hat > 0).all()

    # Compute loss
    ground_truth = torch.randn(batch_size, obs_dim)
    loss_dict = bse.compute_loss(b_hat, sigma_hat, ground_truth)

    assert "loss_bse" in loss_dict
    assert "loss_mse" in loss_dict
    assert "loss_nll" in loss_dict
    assert loss_dict["loss_bse"].item() >= 0


def test_confidence_modulated_lagrangian():
    lagrangian = PIDLagrangianMultiplier(num_constraints=3, use_confidence_modulation=True)

    gamma_high_sigma = lagrangian.compute_gamma(sigma_hat=2.0, budget=50.0)
    gamma_low_sigma = lagrangian.compute_gamma(sigma_hat=0.01, budget=50.0)

    # Higher uncertainty (sigma) -> lower gamma -> stricter constraints
    assert gamma_high_sigma <= gamma_low_sigma

    costs = np.array([2.0, 0.0, 1.0], dtype=np.float32)
    limits = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    lambdas = lagrangian.update(costs, limits, sigma_hat=0.5)

    assert len(lambdas) == 3
    assert (lambdas >= 0).all()


def test_bse_mappo_lce_controller():
    agent_ids = ["intersection_0", "intersection_1"]
    ctrl = BSEMAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4, history_len=3)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)

    assert len(actions) == 2
    assert all(a in [0, 1, 2, 3] for a in actions.values())

    rewards = {aid: -2.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    violations = {"green_time": 1, "green_skip": 0, "phase_skip": 0}
    ctrl.store_transition(rewards, dones, step_violations=violations)

    losses = ctrl.update()
    assert "policy_loss" in losses
    assert "reward_value_loss" in losses
    assert "cost_value_loss" in losses


def test_bse_pretraining_and_calibration():
    env = SUMOEnv(num_intersections=2, max_steps=20, seed=42)
    X, Y = collect_bse_dataset(env, num_steps=20, history_len=3, noise_scenarios=["clean", "mild"])

    assert X.ndim == 3
    assert Y.ndim == 2

    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=16, belief_dim=16, history_len=3)
    loss_history = train_bse_supervised(bse, X, Y, epochs=2, batch_size=8)

    assert len(loss_history) == 2

    calib_metrics = evaluate_bse_calibration(bse, X, Y)
    assert "raw_mse" in calib_metrics
    assert "bse_mse" in calib_metrics
    assert "sigma_calibration_correlation" in calib_metrics
