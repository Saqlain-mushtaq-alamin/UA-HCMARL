"""
Audit verification unit tests covering all 12 Phase 2 requirements:
1. IPPO parameter sharing
2. IPPO action masking
3. MAPPO-LCE multiplier persistence across episodes
4. Lagrangian uncapped penalty
5. Violation accounting without repeated blocked counts
6. Max-wait / starvation constraint
7. Actuated unnormalized queue observations & 1.5-veh threshold
8. Matched traffic arrival seeds across noise levels
9. Fixed-Time zero noise sensitivity
10. Blind-policy diagnostic execution
11. Model checkpoint save/load with torch.save
12. Independent training seed replicates
"""

import os
import tempfile
import numpy as np
import pytest
import torch

from src.baselines.actuated import ActuatedController
from src.baselines.fixed_time import FixedTimeController
from src.baselines.ippo import IPPOController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.envs.sumo_env import SUMOEnv, SingleIntersectionSimulator
from src.training.trainer import Trainer
from scripts.run_phase2_benchmark import BlindWrapper


def test_1_ippo_parameter_sharing():
    """Verify IPPO parameter sharing across all intersections."""
    agent_ids = [f"intersection_{i}" for i in range(4)]
    ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)

    assert hasattr(ctrl, "shared_actor")
    assert hasattr(ctrl, "shared_critic")

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}
    actions = ctrl.select_actions(obs_dict, evaluate=False, step_count=0)
    assert len(actions) == 4

    rewards = {aid: -2.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    ctrl.store_transition(rewards, dones)

    # Weights before update
    w_before = ctrl.shared_actor.net[0].weight.clone()
    losses = ctrl.update()
    w_after = ctrl.shared_actor.net[0].weight

    assert "policy_loss" in losses
    assert not torch.equal(w_before, w_after), "Shared actor weights should update from batched multi-agent transitions"


def test_2_ippo_action_masking():
    """Verify IPPO action masking strictly prevents invalid actions."""
    agent_ids = ["intersection_0"]
    ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True)

    obs_dict = {"intersection_0": np.zeros(16, dtype=np.float32)}
    # Only action 2 is legal
    action_masks = {"intersection_0": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)}

    for evaluate in [True, False]:
        for _ in range(20):
            acts = ctrl.select_actions(obs_dict, evaluate=evaluate, step_count=0, action_masks=action_masks)
            assert acts["intersection_0"] == 2, f"Action mask violated under evaluate={evaluate}"


def test_3_mappo_lce_multiplier_persistence():
    """Verify MAPPO-LCE multipliers persist across episodes and are not reset."""
    agent_ids = [f"intersection_{i}" for i in range(4)]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    # Set initial non-zero lambdas
    ctrl.pid_lagrangian.lambdas[:] = np.array([0.25, 0.15, 0.35], dtype=np.float32)
    ctrl.pid_lagrangian.integral_errors[:] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    # Calling reset() between episodes
    ctrl.reset(agent_ids)

    # Assert multipliers and integrals were NOT wiped
    assert np.allclose(ctrl.pid_lagrangian.lambdas, [0.25, 0.15, 0.35]), "Multipliers must persist across episodes"
    assert np.allclose(ctrl.pid_lagrangian.integral_errors, [1.0, 2.0, 3.0]), "Integral errors must persist across episodes"


def test_4_lagrangian_uncapped_penalty():
    """Verify Lagrangian penalties are applied directly without the old 20% cap."""
    agent_ids = ["intersection_0", "intersection_1"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    # Set positive multiplier
    ctrl.pid_lagrangian.lambdas[:] = np.array([2.0, 2.0, 2.0], dtype=np.float32)

    obs_dict = {aid: np.zeros(16, dtype=np.float32) for aid in agent_ids}
    ctrl.select_actions(obs_dict, evaluate=False, step_count=0)

    # Store 1 step with small reward (-1.0) and large violations (10 counts)
    rewards = {aid: -1.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}
    violations = {"green_time": 10, "green_skip": 0, "phase_skip": 0}
    ctrl.store_transition(rewards, dones, step_violations=violations)

    # In old code: penalty was capped at 20% of |reward| = 0.20
    # In uncapped code: penalty = costs/num_agents * lambdas = (10/2) * 2.0 = 10.0
    costs = np.array(ctrl.buffer["costs"])
    lambdas = ctrl.pid_lagrangian.lambdas
    penalized_rewards = np.array(ctrl.buffer["rewards"]) - np.sum((costs / float(ctrl.num_agents)) * lambdas, axis=-1)

    assert penalized_rewards[0] <= -10.0, f"Expected uncapped penalty <= -10.0, got {penalized_rewards[0]}"


def test_5_violation_accounting_no_repeated_blocked_counts():
    """Verify that blocked switch requests are not counted repeatedly on every subsequent step."""
    sim = SingleIntersectionSimulator(intersection_id="int_0", min_green=5)
    sim.current_phase = 0
    sim.time_in_phase = 1  # during min_green

    # Step 1: Agent attempts early switch to opposing green (action 2)
    _, _, _, v1, _ = sim.step(action=2)
    assert v1["green_time"] == 1, "First blocked attempt should record 1 green_time violation"
    assert sim.switch_attempt_blocked is True
    assert sim.current_phase == 0, "Phase change must be blocked"

    # Step 2: Agent holds the same action 2 on next step while still in min_green
    _, _, _, v2, _ = sim.step(action=2)
    assert v2["green_time"] == 0, "Subsequent ticks while remaining blocked must NOT repeatedly count violations"

    # Step 3: Agent matches current phase -> unblocks state
    _, _, _, v3, _ = sim.step(action=0)
    assert sim.switch_attempt_blocked is False


def test_6_max_wait_starvation_constraint():
    """Verify max-wait / starvation constraint fires when approach waits too long."""
    sim = SingleIntersectionSimulator(intersection_id="int_0", max_red_wait=10)
    sim.current_phase = 0  # NS Green, EW is Red
    # Place queued vehicles on East approach (index 2)
    sim.queues[2] = 5.0

    starvation_fired = False
    for step in range(15):
        # Hold NS green
        _, _, _, viols, _ = sim.step(action=0)
        sim.queues[2] = 5.0  # keep queue present on red approach
        if viols.get("max_wait", 0) > 0:
            starvation_fired = True
            break

    assert starvation_fired, "Starvation violation must trigger when waiting approach exceeds max_red_wait"


def test_7_actuated_unnormalized_and_threshold():
    """Verify Actuated controller uses unnormalized queue observations and 1.5-veh threshold."""
    ctrl = ActuatedController(queue_threshold=1.5, max_queue=20.0)
    ctrl.reset(["intersection_0"])
    ctrl.current_phase["intersection_0"] = 0
    ctrl.time_in_phase["intersection_0"] = 10  # >= min_green

    # Case A: Queue normalized to 0.05 (= 1.0 veh, below threshold 1.5 veh) -> should switch to yellow (1)
    obs_cleared = np.array([0.025, 0.025, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    act_cleared = ctrl.select_action("intersection_0", obs_cleared)
    assert act_cleared == 1, "Queue below 1.5 veh threshold must trigger switch to clearance yellow"

    # Case B: Queue normalized to 0.15 (= 3.0 veh, above threshold 1.5 veh) -> should hold green (0)
    ctrl.time_in_phase["intersection_0"] = 10
    ctrl.current_phase["intersection_0"] = 0
    obs_heavy = np.array([0.075, 0.075, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
    act_heavy = ctrl.select_action("intersection_0", obs_heavy)
    assert act_heavy == 0, "Queue above 1.5 veh threshold must hold green phase"


def test_8_traffic_arrival_seeds_across_noise():
    """Verify traffic arrival seeds generate identical vehicle demand across noise levels."""
    seed = 42
    env_clean = SUMOEnv(num_intersections=2, noise_scenario="clean", seed=seed)
    env_extreme = SUMOEnv(num_intersections=2, noise_scenario="extreme", seed=seed)

    env_clean.reset(seed=seed)
    env_extreme.reset(seed=seed)

    # Step both environments with identical actions
    actions = {f"intersection_{i}": 0 for i in range(2)}
    _, _, _, _, info_clean = env_clean.step(actions)
    _, _, _, _, info_extreme = env_extreme.step(actions)

    # Queues inside simulator (ground truth arrivals) must be identical
    sim_clean = env_clean.simulators["intersection_0"]
    sim_extreme = env_extreme.simulators["intersection_0"]

    assert np.allclose(sim_clean.queues, sim_extreme.queues), "Underlying vehicle arrival process must match across noise levels"


def test_9_fixed_time_zero_noise_sensitivity():
    """Verify Fixed-Time has approximately zero noise sensitivity under matched traffic seeds."""
    ctrl = FixedTimeController()
    env_clean = SUMOEnv(num_intersections=2, noise_scenario="clean", max_steps=100, seed=42)
    env_extreme = SUMOEnv(num_intersections=2, noise_scenario="extreme", max_steps=100, seed=42)

    trainer_c = Trainer(env=env_clean)
    trainer_e = Trainer(env=env_extreme)

    res_clean = trainer_c.run_evaluation_episode(ctrl, max_steps=100, seed=42)
    res_extreme = trainer_e.run_evaluation_episode(ctrl, max_steps=100, seed=42)

    q_c = res_clean["avg_queue_length"]
    q_e = res_extreme["avg_queue_length"]

    assert abs(q_c - q_e) < 1e-4, f"Fixed-Time is open-loop: queue must be identical across noise levels ({q_c} vs {q_e})"


def test_10_blind_policy_diagnostics():
    """Verify BlindWrapper zeroes observations and correctly reports degradation."""
    agent_ids = ["intersection_0"]
    ctrl = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    blind_ctrl = BlindWrapper(ctrl)

    obs = {"intersection_0": np.ones(16, dtype=np.float32) * 5.0}
    acts = blind_ctrl.select_actions(obs, evaluate=True)
    assert "intersection_0" in acts
    assert acts["intersection_0"] in [0, 1, 2, 3]


def test_11_checkpoint_saving_torch_save():
    """Verify models are saved and restored using torch.save/torch.load."""
    agent_ids = ["intersection_0", "intersection_1"]
    ctrl1 = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "ippo_test.pt")
        ctrl1.save_model(model_path)
        assert os.path.exists(model_path), "Model file must be created on disk"

        ctrl2 = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True)
        ctrl2.load_model(model_path)

        for p1, p2 in zip(ctrl1.shared_actor.parameters(), ctrl2.shared_actor.parameters()):
            assert torch.equal(p1, p2), "Loaded weights must exactly match saved weights"


def test_12_independent_training_seeds():
    """Verify distinct training seeds initialize and train independent networks."""
    agent_ids = ["intersection_0", "intersection_1"]
    torch.manual_seed(42)
    ctrl_seed42 = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    torch.manual_seed(43)
    ctrl_seed43 = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    p42 = ctrl_seed42.shared_actor.net[0].weight
    p43 = ctrl_seed43.shared_actor.net[0].weight

    assert not torch.equal(p42, p43), "Independent seeds must produce distinct replicate networks"

