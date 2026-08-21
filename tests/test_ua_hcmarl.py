"""
Unit tests for Layer 3 Uncertainty-Aware Manager & Complete UA-HCMARL Controller (Phase 4)
"""

import pytest
import numpy as np
import torch
from src.models.lp_projection import LPBudgetProjection, PyTorchLPProjectionLayer
from src.models.manager_agent import UncertaintyAwareManager, compute_manager_state
from src.baselines.ua_hcmarl import UAHCMARLController


def test_lp_projection():
    proj = LPBudgetProjection(num_workers=4, b_total=100.0, b_min=5.0, b_max=50.0)
    raw = np.array([80.0, 10.0, 5.0, 5.0])
    res = proj.project(raw)

    assert len(res) == 4
    assert np.isclose(np.sum(res), 100.0, atol=1e-3)
    assert (res >= 5.0 - 1e-3).all()
    assert (res <= 50.0 + 1e-3).all()


def test_pytorch_lp_projection_layer():
    layer = PyTorchLPProjectionLayer(num_workers=4, b_total=100.0, b_min=5.0, b_max=50.0)
    raw_tensor = torch.tensor([80.0, 10.0, 5.0, 5.0], requires_grad=True)

    projected = layer(raw_tensor)
    assert torch.isclose(projected.sum(), torch.tensor(100.0), atol=1e-3)

    # Test backward pass
    loss = projected.sum()
    loss.backward()
    assert raw_tensor.grad is not None


def test_manager_agent():
    mgr = UncertaintyAwareManager(num_workers=4, state_dim=21, hidden_dim=32, b_total=100.0)
    state = torch.randn(1, 21)

    dist, budgets, value = mgr(state)

    assert budgets.shape == (1, 4)
    assert torch.isclose(budgets.sum(), torch.tensor(100.0), atol=1e-2)
    assert value.shape == (1,)


def test_ua_hcmarl_controller():
    agent_ids = [f"intersection_{i}" for i in range(16)]
    ctrl = UAHCMARLController(agent_ids=agent_ids, obs_dim=16, action_dim=4, t_manager=15)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)

    assert len(actions) == 16
    assert all(a in [0, 1, 2, 3] for a in actions.values())

    rewards = {aid: -1.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    violations = {"green_time": 0, "green_skip": 0, "phase_skip": 0}
    ctrl.store_transition(rewards, dones, step_violations=violations)

    losses = ctrl.update()
    assert "worker_policy_loss" in losses
    assert "manager_loss" in losses
