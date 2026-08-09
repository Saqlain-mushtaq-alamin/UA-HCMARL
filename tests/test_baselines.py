"""
Unit tests for src/baselines/fixed_time.py and src/baselines/actuated.py
"""

import pytest
import numpy as np
from src.baselines.fixed_time import FixedTimeController
from src.baselines.actuated import ActuatedController
from src.envs.sumo_env import SUMOEnv


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
    
    obs = np.array([5.0, 5.0, 0.0, 0.0] + [0.0]*12, dtype=np.float32)
    action = ctrl.select_action("intersection_0", obs)
    assert action in [0, 1, 2, 3]


def test_baselines_in_env():
    env = SUMOEnv(num_intersections=2, max_steps=20, seed=42)
    obs_dict, _ = env.reset(seed=42)
    
    ft_ctrl = FixedTimeController()
    actions = ft_ctrl.select_actions(obs_dict, step_count=0)
    next_obs, rewards, dones, truncateds, info = env.step(actions)
    
    assert len(next_obs) == 2
