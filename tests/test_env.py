"""
Unit tests for src/envs/sumo_env.py and src/envs/multi_agent_env.py
"""

import pytest
import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.envs.multi_agent_env import MARLTSCParallelEnv


def test_sumo_env_reset_step():
    env = SUMOEnv(num_intersections=4, max_steps=10, seed=42)
    obs_dict, info = env.reset(seed=42)
    
    assert len(obs_dict) == 4
    for aid, obs in obs_dict.items():
        assert obs.shape == (16,)
        assert np.all(obs >= 0.0)

    actions = {aid: 0 for aid in obs_dict.keys()}
    next_obs, rewards, dones, truncateds, next_info = env.step(actions)
    
    assert len(next_obs) == 4
    assert len(rewards) == 4
    assert "step_delays" in next_info
    assert "step_queues" in next_info
    assert "step_throughput" in next_info


def test_pettingzoo_parallel_env():
    p_env = MARLTSCParallelEnv(num_intersections=4, max_steps=5, seed=42)
    obs_dict, info = p_env.reset(seed=42)
    
    assert len(p_env.agents) == 4
    actions = {agent: p_env.action_space(agent).sample() for agent in p_env.agents}
    
    next_obs, rewards, dones, truncateds, next_info = p_env.step(actions)
    assert len(next_obs) == 4
    assert len(rewards) == 4
