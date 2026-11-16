"""
Controller Isolation & Order-Invariance Tests (Phase 3 Blocker B3)
==================================================================
Verifies that controllers across experimental arms (M1 vs M3, M4 vs M6)
use separate in-memory object instances and produce deterministic, order-invariant
decisions without cross-arm state pollution.
"""

import pytest
import numpy as np
import copy

from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_wrapper import BSEControllerWrapper
from src.models.belief_encoder import GRUKalmanNet
from src.training.trainer import Trainer


def test_independent_instances_for_bse_arms():
    """Verify that M3 and M6 use distinct controller instances from M1 and M4."""
    agent_ids = [f"intersection_{i}" for i in range(4)]
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)

    # Clean IPPO (M1) and IPPO Base for BSE (M3)
    m1_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    m3_ippo_base = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    m3_bse = BSEControllerWrapper(base_controller=m3_ippo_base, bse_model=bse, history_len=3)

    assert m1_ippo is not m3_bse.base_controller, "M1 and M3 must not share the same controller instance!"

    # Clean FMA2C (M4) and FMA2C Base for BSE (M6)
    m4_fma2c = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=2)
    m6_fma2c_base = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=2)
    m6_bse = BSEControllerWrapper(base_controller=m6_fma2c_base, bse_model=bse, history_len=3)

    assert m4_fma2c is not m6_bse.base_controller, "M4 and M6 must not share the same controller instance!"


def test_order_invariance_across_arms():
    """
    B3 Verification:
    Proves that evaluating arms in different orders (M4 then M6 vs M6 then M4)
    produces identical results under identical inputs and seeds.
    """
    agent_ids = [f"intersection_{i}" for i in range(4)]
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)

    # Initial controllers with identical initial weights
    m4_ctrl = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=2)
    m6_base = copy.deepcopy(m4_ctrl)
    m6_ctrl = BSEControllerWrapper(base_controller=m6_base, bse_model=bse, history_len=3)

    env_seed = 1000
    max_steps = 30

    # Trial 1: Evaluate M4 first, then M6
    env1 = SUMOEnv(num_intersections=4, noise_scenario="moderate", max_steps=max_steps, seed=env_seed)
    trainer1 = Trainer(env=env1)

    m4_ctrl.reset(agent_ids)
    summary_m4_trial1 = trainer1.run_evaluation_episode(m4_ctrl, max_steps=max_steps, seed=env_seed)

    m6_ctrl.reset(agent_ids)
    summary_m6_trial1 = trainer1.run_evaluation_episode(m6_ctrl, max_steps=max_steps, seed=env_seed)

    # Trial 2: Evaluate M6 first, then M4 (reversed order)
    env2 = SUMOEnv(num_intersections=4, noise_scenario="moderate", max_steps=max_steps, seed=env_seed)
    trainer2 = Trainer(env=env2)

    m6_ctrl.reset(agent_ids)
    summary_m6_trial2 = trainer2.run_evaluation_episode(m6_ctrl, max_steps=max_steps, seed=env_seed)

    m4_ctrl.reset(agent_ids)
    summary_m4_trial2 = trainer2.run_evaluation_episode(m4_ctrl, max_steps=max_steps, seed=env_seed)

    # Assert complete order invariance
    assert summary_m4_trial1["avg_delay"] == summary_m4_trial2["avg_delay"], \
        f"M4 results varied by evaluation order! ({summary_m4_trial1['avg_delay']} vs {summary_m4_trial2['avg_delay']})"

    assert summary_m6_trial1["avg_delay"] == summary_m6_trial2["avg_delay"], \
        f"M6 results varied by evaluation order! ({summary_m6_trial1['avg_delay']} vs {summary_m6_trial2['avg_delay']})"

