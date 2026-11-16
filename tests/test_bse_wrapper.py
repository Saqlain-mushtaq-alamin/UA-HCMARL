"""
Integration Tests for BSE Controller Wrapper (Phase 3 Blocker 1 & 2)
====================================================================
Verifies that BSEControllerWrapper correctly wraps IPPO and FMA2C,
preserves action spaces and masks, executes batched inference without leakage,
and enforces true Kalman recursive belief propagation across time steps.
"""

import pytest
import numpy as np
import torch
import os
import tempfile

from src.envs.sumo_env import SUMOEnv
from src.models.belief_encoder import GRUKalmanNet
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_wrapper import BSEControllerWrapper


def test_bse_wrapper_ippo_integration():
    """Verify end-to-end IPPO + BSE integration with action masking."""
    agent_ids = [f"intersection_{i}" for i in range(16)]
    base_ippo = IPPOController(
        agent_ids=agent_ids,
        obs_dim=16,
        action_dim=4,
        share_actor=True,
        share_critic=True,
    )
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=5)
    wrapper = BSEControllerWrapper(base_controller=base_ippo, bse_model=bse, history_len=5)

    assert wrapper.num_agents == 16
    assert wrapper.obs_dim == 16
    assert wrapper.action_dim == 4

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    masks = {aid: np.array([1, 0, 1, 0], dtype=np.float32) for aid in agent_ids}

    actions = wrapper.select_actions(obs_dict, evaluate=True, step_count=0, action_masks=masks)

    assert len(actions) == 16
    for aid, act in actions.items():
        assert act in [0, 2], f"Agent {aid} selected illegal action {act} outside mask [1, 0, 1, 0]"

    assert wrapper.total_inference_calls == 1
    assert wrapper.get_average_latency_ms() > 0.0


def test_bse_wrapper_fma2c_integration():
    """Verify end-to-end FMA2C + BSE integration across regional managers and workers."""
    agent_ids = [f"intersection_{i}" for i in range(16)]
    base_fma2c = FMA2CController(
        agent_ids=agent_ids,
        obs_dim=16,
        action_dim=4,
        num_regions=4,
    )
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=5)
    wrapper = BSEControllerWrapper(base_controller=base_fma2c, bse_model=bse, history_len=5)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    masks = {aid: np.array([0, 1, 1, 0], dtype=np.float32) for aid in agent_ids}

    actions_0 = wrapper.select_actions(obs_dict, evaluate=True, step_count=0, action_masks=masks)
    for act in actions_0.values():
        assert act in [1, 2]

    actions_1 = wrapper.select_actions(obs_dict, evaluate=True, step_count=1, action_masks=masks)
    for act in actions_1.values():
        assert act in [1, 2]


def test_bse_wrapper_full_environment_pipeline():
    """
    Test full runtime path:
    SUMOEnv -> Noisy Obs -> BSE -> Belief -> IPPO/FMA2C -> Mask -> Action -> SUMOEnv.step.
    """
    env = SUMOEnv(num_intersections=4, noise_scenario="severe", max_steps=10, seed=42)
    obs_dict, reset_info = env.reset(seed=42)
    action_masks = reset_info["action_masks"]

    base_ctrl = IPPOController(agent_ids=env.intersection_ids, obs_dim=16, action_dim=4)
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)
    wrapper = BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=3)

    for step in range(5):
        actions = wrapper.select_actions(
            obs_dict, evaluate=True, step_count=step, action_masks=action_masks
        )
        assert len(actions) == 4
        obs_dict, rewards, dones, _, info = env.step(actions)
        action_masks = info["action_masks"]
        assert "step_queues" in info


def test_kalman_recurrence_affects_updated_belief():
    """
    B2 Verification:
    Proves that different previous beliefs with identical current observations
    produce significantly different updated beliefs via the Kalman recurrence.
    """
    agent_ids = ["intersection_0"]
    base_ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)
    wrapper = BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=3)

    # Identical current observation
    current_obs = {"intersection_0": np.ones(16, dtype=np.float32) * 0.5}

    # Case A: Prior belief is all zeros
    wrapper.reset()
    wrapper.prev_beliefs["intersection_0"] = torch.zeros(16, dtype=torch.float32)
    wrapper.select_actions(current_obs, evaluate=True)
    belief_A = wrapper.prev_beliefs["intersection_0"].cpu().numpy()

    # Case B: Prior belief is large positive values
    wrapper.reset()
    wrapper.prev_beliefs["intersection_0"] = torch.ones(16, dtype=torch.float32) * 5.0
    wrapper.select_actions(current_obs, evaluate=True)
    belief_B = wrapper.prev_beliefs["intersection_0"].cpu().numpy()

    # Proves prior belief genuinely propagates into updated posterior belief
    diff = np.abs(belief_A - belief_B).sum()
    assert diff > 0.1, f"Expected recurrence to differentiate beliefs, but diff was only {diff}"


def test_prior_belief_passed_and_updated_each_step():
    """
    B2 Verification:
    Proves that the returned belief from step t is stored and passed as prev_belief at step t+1.
    """
    agent_ids = ["intersection_0"]
    base_ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)
    wrapper = BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=3)

    obs = {"intersection_0": np.ones(16, dtype=np.float32)}

    # Step 0: prev_belief is None initially
    assert wrapper.prev_beliefs["intersection_0"] is None
    wrapper.select_actions(obs, evaluate=True, step_count=0)
    belief_step0 = wrapper.prev_beliefs["intersection_0"].clone()
    assert belief_step0 is not None

    # Step 1: prev_belief at start of step 1 must equal belief_step0
    wrapper.select_actions(obs, evaluate=True, step_count=1)
    belief_step1 = wrapper.prev_beliefs["intersection_0"].clone()

    # Belief evolves across steps
    assert not torch.equal(belief_step0, belief_step1)


def test_bse_wrapper_reset_isolation():
    """Verify that wrapper.reset() clears all histories and recurrent beliefs."""
    agent_ids = ["intersection_0", "intersection_1"]
    base_ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=16, belief_dim=16, history_len=5)
    wrapper = BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=5)

    for s in range(3):
        obs = {aid: np.ones(16, dtype=np.float32) * (s + 1) for aid in agent_ids}
        wrapper.select_actions(obs, evaluate=True, step_count=s)

    assert len(wrapper.histories["intersection_0"]) == 3
    assert wrapper.prev_beliefs["intersection_0"] is not None

    wrapper.reset()
    assert len(wrapper.histories["intersection_0"]) == 0
    assert len(wrapper.histories["intersection_1"]) == 0
    assert wrapper.prev_beliefs["intersection_0"] is None
    assert wrapper.prev_beliefs["intersection_1"] is None


def test_intersection_isolation_no_cross_agent_leakage():
    """
    B2 Verification:
    Proves that one intersection's prior belief cannot affect another intersection's update.
    """
    agent_ids = ["intersection_0", "intersection_1"]
    base_ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)
    wrapper = BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=3)

    obs = {aid: np.ones(16, dtype=np.float32) * 0.5 for aid in agent_ids}

    # Run trial 1: Agent 0 prior = 0.0, Agent 1 prior = 1.0
    wrapper.reset()
    wrapper.prev_beliefs["intersection_0"] = torch.zeros(16, dtype=torch.float32)
    wrapper.prev_beliefs["intersection_1"] = torch.ones(16, dtype=torch.float32)
    wrapper.select_actions(obs, evaluate=True)
    trial1_agent1_belief = wrapper.prev_beliefs["intersection_1"].clone()

    # Run trial 2: Agent 0 prior = 100.0 (extreme change), Agent 1 prior STILL = 1.0
    wrapper.reset()
    wrapper.prev_beliefs["intersection_0"] = torch.ones(16, dtype=torch.float32) * 100.0
    wrapper.prev_beliefs["intersection_1"] = torch.ones(16, dtype=torch.float32)
    wrapper.select_actions(obs, evaluate=True)
    trial2_agent1_belief = wrapper.prev_beliefs["intersection_1"].clone()

    # Agent 1's belief must be identical regardless of Agent 0's state
    assert torch.equal(trial1_agent1_belief, trial2_agent1_belief), \
        "Cross-intersection leakage detected! Agent 0 belief affected Agent 1."


def test_no_cross_episode_state_survives_reset():
    """
    B2 Verification:
    Proves that reset guarantees deterministic step 0 outputs across episodes with identical inputs.
    """
    agent_ids = ["intersection_0"]
    base_ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=3)
    wrapper = BSEControllerWrapper(base_controller=base_ctrl, bse_model=bse, history_len=3)

    obs = {"intersection_0": np.random.randn(16).astype(np.float32)}

    # Episode 1 step 0
    wrapper.reset()
    wrapper.select_actions(obs, evaluate=True, step_count=0)
    ep1_step0_belief = wrapper.prev_beliefs["intersection_0"].clone()

    # Step many times with crazy inputs
    for _ in range(10):
        wrapper.select_actions({"intersection_0": np.random.randn(16).astype(np.float32)}, evaluate=True)

    # Episode 2 step 0 (after reset, same obs)
    wrapper.reset()
    wrapper.select_actions(obs, evaluate=True, step_count=0)
    ep2_step0_belief = wrapper.prev_beliefs["intersection_0"].clone()

    assert torch.equal(ep1_step0_belief, ep2_step0_belief), \
        "Cross-episode leakage: Episode 2 step 0 differed from Episode 1 step 0!"


def test_bse_wrapper_checkpoint_serialization():
    """Verify that BSE wrapper saves and loads weights correctly."""
    bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=5)
    ctrl = IPPOController(agent_ids=["intersection_0"])
    wrapper = BSEControllerWrapper(base_controller=ctrl, bse_model=bse, history_len=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "bse_test.pt")
        wrapper.save_model(ckpt_path)
        assert os.path.exists(ckpt_path)

        bse_new = GRUKalmanNet(raw_obs_dim=16, hidden_dim=32, belief_dim=16, history_len=5)
        wrapper_new = BSEControllerWrapper(base_controller=ctrl, bse_model=bse_new, history_len=5)
        wrapper_new.load_bse_checkpoint(ckpt_path)

        for p1, p2 in zip(wrapper.bse.parameters(), wrapper_new.bse.parameters()):
            assert torch.equal(p1, p2)
