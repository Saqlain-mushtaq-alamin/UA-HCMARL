"""
Unit tests for all Stage 0-2 Baselines (Fixed-Time, Actuated, IPPO, MAPPO, MAPPO-LCE, FMA2C)
"""

import pytest
import numpy as np
from src.baselines.fixed_time import FixedTimeController
from src.baselines.actuated import ActuatedController
from src.baselines.ippo import IPPOController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.baselines.fma2c import FMA2CController
from src.envs.sumo_env import SUMOEnv
from src.training.trainer import Trainer


def test_fixed_time_controller():
    ctrl = FixedTimeController(green_duration=10, yellow_duration=2)
    obs = np.zeros(16, dtype=np.float32)

    assert ctrl.select_action(obs, step_count=0) == 0
    assert ctrl.select_action(obs, step_count=5) == 0
    assert ctrl.select_action(obs, step_count=10) == 1
    assert ctrl.select_action(obs, step_count=12) == 2


def test_actuated_controller():
    ctrl = ActuatedController(min_green=5, max_green=20, yellow_duration=3)
    ctrl.reset(["intersection_0"])

    obs = np.array([5.0, 5.0, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    action = ctrl.select_action("intersection_0", obs)
    assert action in [0, 1, 2, 3]


def test_baselines_in_env():
    env = SUMOEnv(num_intersections=2, max_steps=20, seed=42)
    obs_dict, _ = env.reset(seed=42)

    ft_ctrl = FixedTimeController()
    actions = ft_ctrl.select_actions(obs_dict, step_count=0)
    next_obs, rewards, dones, truncateds, info = env.step(actions)

    assert len(next_obs) == 2


def test_ippo_controller():
    agent_ids = ["intersection_0", "intersection_1"]
    ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)
    assert len(actions) == 2
    assert all(a in [0, 1, 2, 3] for a in actions.values())

    rewards = {aid: -5.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    ctrl.store_transition(rewards, dones)

    losses = ctrl.update()
    assert "policy_loss" in losses


def test_mappo_controller():
    agent_ids = ["intersection_0", "intersection_1"]
    ctrl = MAPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)
    assert len(actions) == 2

    rewards = {aid: -3.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    ctrl.store_transition(rewards, dones)

    losses = ctrl.update()
    assert "policy_loss" in losses
    assert "value_loss" in losses


def test_mappo_lce_controller():
    agent_ids = ["intersection_0", "intersection_1"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)
    assert len(actions) == 2

    rewards = {aid: -4.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    violations = {"green_time": 1, "green_skip": 0, "phase_skip": 0}
    ctrl.store_transition(rewards, dones, step_violations=violations)

    losses = ctrl.update()
    assert "policy_loss" in losses
    assert "reward_value_loss" in losses
    assert "lambda_green" in losses


def test_fma2c_controller():
    agent_ids = [f"intersection_{i}" for i in range(4)]
    ctrl = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=2)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)
    assert len(actions) == 4

    rewards = {aid: -2.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    ctrl.store_transition(rewards, dones)

    losses = ctrl.update()
    assert "worker_loss" in losses
    assert "manager_loss" in losses
