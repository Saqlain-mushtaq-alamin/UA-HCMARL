"""
Deterministic unit tests for MAPPO-LCE starvation constraint integration (Task 2).

Tests:
1. Four-constraint initialization and default budget configuration.
2. Zero starvation cost does not increase lambda.
3. Positive starvation cost increases lambda.
4. Multipliers remain non-negative and bounded.
5. Multiplier values persist across episode boundaries (reset does not clear lambdas).
6. Checkpoint serialization and restoration preserves all 4 multipliers and PID states.
"""

import os
import numpy as np
import torch
import pytest

from src.baselines.mappo_lce import MAPPOLCEController
from src.envs.sumo_env import SingleIntersectionSimulator, SUMOEnv


def test_mappo_lce_4_constraints_initialization():
    """Verify MAPPO-LCE initializes with 4 constraint dimensions and valid default budgets."""
    agent_ids = ["int_0", "int_1"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    assert ctrl.num_constraints == 4, f"Expected 4 constraints, got {ctrl.num_constraints}"
    assert len(ctrl.constraint_limits) == 4, f"Expected 4 limits, got {len(ctrl.constraint_limits)}"
    assert len(ctrl.pid_lagrangian.lambdas) == 4, f"Expected 4 lambdas, got {len(ctrl.pid_lagrangian.lambdas)}"
    assert np.all(ctrl.pid_lagrangian.lambdas == 0.0), "Initial lambdas must be 0.0"
    # Check default limits: [0.3, 0.3, 0.3, 0.01]
    assert np.allclose(ctrl.constraint_limits, [0.3, 0.3, 0.3, 0.01]), f"Unexpected default limits: {ctrl.constraint_limits}"


def test_mappo_lce_zero_starvation_cost_does_not_increase_lambda():
    """Verify zero starvation cost does not increase lambda."""
    agent_ids = ["int_0", "int_1"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    # Populate buffer with transitions where all violations are zero
    obs_dict = {aid: np.zeros(16, dtype=np.float32) for aid in agent_ids}
    rewards = {aid: -1.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}

    for _ in range(10):
        ctrl.select_actions(obs_dict, evaluate=False)
        ctrl.store_transition(
            rewards=rewards,
            dones=dones,
            step_violations={"green_time": 0, "green_skip": 0, "phase_skip": 0, "max_wait": 0},
        )

    loss_dict = ctrl.update(num_epochs=1, batch_size=4)
    assert ctrl.pid_lagrangian.lambdas[3] == 0.0, "Lambda starve must remain 0.0 when starvation cost is zero"
    assert loss_dict["lambda_starve"] == 0.0


def test_mappo_lce_positive_starvation_cost_increases_lambda():
    """Verify positive starvation cost strictly increases lambda."""
    agent_ids = ["int_0", "int_1"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    obs_dict = {aid: np.zeros(16, dtype=np.float32) for aid in agent_ids}
    rewards = {aid: -5.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}

    # Store transitions with heavy starvation violations (both agents violating max_wait)
    for _ in range(20):
        ctrl.select_actions(obs_dict, evaluate=False)
        ctrl.store_transition(
            rewards=rewards,
            dones=dones,
            step_violations={"green_time": 0, "green_skip": 0, "phase_skip": 0, "max_wait": 2},
        )

    loss_dict = ctrl.update(num_epochs=1, batch_size=4)
    lambda_starve = ctrl.pid_lagrangian.lambdas[3]

    assert lambda_starve > 0.0, f"Lambda starve must increase on positive constraint violation, got {lambda_starve}"
    assert loss_dict["lambda_starve"] == lambda_starve


def test_mappo_lce_multiplier_non_negative_and_bounded():
    """Verify multipliers remain strictly non-negative and respect bounds."""
    agent_ids = ["int_0"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    obs_dict = {aid: np.zeros(16, dtype=np.float32) for aid in agent_ids}
    rewards = {aid: 0.0 for aid in agent_ids}
    dones = {aid: False for aid in agent_ids}

    # First update with zero violations -> should remain 0
    for _ in range(5):
        ctrl.select_actions(obs_dict, evaluate=False)
        ctrl.store_transition(
            rewards=rewards,
            dones=dones,
            step_violations={"green_time": 0, "green_skip": 0, "phase_skip": 0, "max_wait": 0},
        )
    ctrl.update(num_epochs=1, batch_size=2)
    assert np.all(ctrl.pid_lagrangian.lambdas >= 0.0), "Multipliers must be non-negative"

    # Now verify max_lambda clamping
    ctrl.pid_lagrangian.lambdas[:] = ctrl.pid_lagrangian.max_lambda - 0.01
    for _ in range(5):
        ctrl.select_actions(obs_dict, evaluate=False)
        ctrl.store_transition(
            rewards=rewards,
            dones=dones,
            step_violations={"green_time": 1, "green_skip": 1, "phase_skip": 1, "max_wait": 1},
        )
    ctrl.update(num_epochs=1, batch_size=2)
    assert np.all(ctrl.pid_lagrangian.lambdas <= ctrl.pid_lagrangian.max_lambda), "Multipliers must not exceed max_lambda"


def test_mappo_lce_persistence_across_episodes():
    """Verify multiplier values persist across episode boundaries when reset() is called."""
    agent_ids = ["int_0", "int_1"]
    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    # Set lambdas to known non-zero values
    ctrl.pid_lagrangian.lambdas[:] = [0.12, 0.34, 0.56, 0.78]
    ctrl.pid_lagrangian.integral_errors[:] = [0.01, 0.02, 0.03, 0.04]

    # Episode boundary: reset()
    ctrl.reset(agent_ids)

    # Verify buffers are empty but PID states persist
    assert len(ctrl.buffer["rewards"]) == 0
    assert len(ctrl.buffer["global_states"]) == 0
    assert np.allclose(ctrl.pid_lagrangian.lambdas, [0.12, 0.34, 0.56, 0.78]), "Lambdas must persist across episodes"
    assert np.allclose(ctrl.pid_lagrangian.integral_errors, [0.01, 0.02, 0.03, 0.04]), "Integral errors must persist"


def test_mappo_lce_checkpoint_serialization_and_restoration(tmp_path):
    """Verify checkpoint save and load preserves all 4 multipliers and PID states."""
    agent_ids = ["int_0", "int_1"]
    ctrl1 = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)

    test_lambdas = np.array([0.15, 0.25, 0.35, 0.85], dtype=np.float32)
    test_ie = np.array([0.05, 0.06, 0.07, 0.08], dtype=np.float32)
    test_pe = np.array([0.01, 0.02, 0.03, 0.04], dtype=np.float32)

    ctrl1.pid_lagrangian.lambdas = np.copy(test_lambdas)
    ctrl1.pid_lagrangian.integral_errors = np.copy(test_ie)
    ctrl1.pid_lagrangian.prev_errors = np.copy(test_pe)

    ckpt_path = os.path.join(tmp_path, "mappo_lce_test.pt")
    ctrl1.save_model(ckpt_path)

    assert os.path.exists(ckpt_path)

    # Restore in new controller instance
    ctrl2 = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    assert np.all(ctrl2.pid_lagrangian.lambdas == 0.0)

    ctrl2.load_model(ckpt_path)

    assert np.allclose(ctrl2.pid_lagrangian.lambdas, test_lambdas), "Restored lambdas must match saved lambdas"
    assert np.allclose(ctrl2.pid_lagrangian.integral_errors, test_ie), "Restored integral errors must match"
    assert np.allclose(ctrl2.pid_lagrangian.prev_errors, test_pe), "Restored prev errors must match"


def test_mappo_lce_legacy_3d_checkpoint_backward_compatibility(tmp_path):
    """Verify loading an older 3-dimensional checkpoint does not crash or corrupt 4D controller."""
    agent_ids = ["int_0", "int_1"]
    ckpt_path = os.path.join(tmp_path, "legacy_3d.pt")

    # Save a legacy checkpoint with only 3 constraints
    legacy_data = {
        "share_actor": True,
        "shared_actor": None,
        "actors": None,
        "pid_lambdas": np.array([0.11, 0.22, 0.33], dtype=np.float32),
    }
    torch.save(legacy_data, ckpt_path)

    ctrl = MAPPOLCEController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_constraints=4)
    ctrl.load_model(ckpt_path)

    # First 3 lambdas restored; 4th (starvation) remains 0.0
    assert np.allclose(ctrl.pid_lagrangian.lambdas[:3], [0.11, 0.22, 0.33])
    assert ctrl.pid_lagrangian.lambdas[3] == 0.0
    assert ctrl.num_constraints == 4


def test_sumo_env_starvation_signal_pipeline():
    """Verify SingleIntersectionSimulator and SUMOEnv correctly track and propagate max_wait."""
    env = SUMOEnv(num_intersections=2, max_steps=100, max_red_wait=38, seed=42)
    obs, info = env.reset(seed=42)

    # In simulator 0, manually inject high red wait on approach 2
    env.simulators["intersection_0"].current_phase = 0  # NS Green, EW is Red
    env.simulators["intersection_0"].queues[2] = 5.0
    env.simulators["intersection_0"].approach_red_wait[2] = 37.0  # 1 step away from starvation

    # Step: action 0 holds NS Green
    actions = {"intersection_0": 0, "intersection_1": 0}
    obs, rewards, dones, truncateds, info = env.step(actions)

    # Approach 2 red wait reaches 38.0 -> max_wait violation triggers!
    assert info["step_violations"]["max_wait"] >= 1, "max_wait violation must propagate to env info"
    assert "step_violations_per_agent" in info
    assert info["step_violations_per_agent"]["intersection_0"]["max_wait"] >= 1

