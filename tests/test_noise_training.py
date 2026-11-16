"""
Unit & Integration Tests for Noise-Trained Baselines (Phase 3 Blocker 2)
========================================================================
Proves that observation noise is actively injected during training,
verifies multi-noise sampling, and tests checkpoint creation for IPPO and FMA2C.
"""

import pytest
import numpy as np
import os
import tempfile

from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.training.noise_training import train_noise_augmented_controller, CANONICAL_NOISE_SCENARIOS


def test_noise_injection_during_training_step():
    """Verify that observation noise is genuinely applied during training steps."""
    env = SUMOEnv(num_intersections=2, noise_scenario="extreme", max_steps=10, seed=42)
    obs_dict, _ = env.reset(seed=42)

    # For extreme noise (d1=0.5, d2=0.3, d3=0.3), noisy obs must differ from clean obs
    clean_obs = env._get_clean_observation("intersection_0")
    noisy_obs = obs_dict["intersection_0"]

    diff = np.abs(noisy_obs - clean_obs).mean()
    assert diff > 0.05, f"Expected non-trivial noise injection, got diff={diff}"


def test_noise_training_curriculum_execution():
    """Verify that train_noise_augmented_controller cycles through noise scenarios."""
    env = SUMOEnv(num_intersections=2, max_steps=20, seed=42)
    ctrl = IPPOController(agent_ids=env.intersection_ids, obs_dim=16, action_dim=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ippo_noise_test.pt")
        curve_path = os.path.join(tmpdir, "curves.json")

        result = train_noise_augmented_controller(
            controller=ctrl,
            env=env,
            train_episodes=5,
            max_steps=20,
            seed=42,
            noise_scenarios=CANONICAL_NOISE_SCENARIOS,
            checkpoint_path=ckpt_path,
            log_curve_path=curve_path,
            verbose=False,
        )

        assert len(result["scenarios_sampled"]) == 5
        assert all(s in CANONICAL_NOISE_SCENARIOS for s in result["scenarios_sampled"])
        assert os.path.exists(ckpt_path)
        assert os.path.exists(curve_path)


def test_fma2c_noise_training():
    """Verify that FMA2C trains under noise without dimension or coordination errors."""
    env = SUMOEnv(num_intersections=4, max_steps=20, seed=43)
    ctrl = FMA2CController(agent_ids=env.intersection_ids, obs_dim=16, action_dim=4, num_regions=2)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "fma2c_noise_test.pt")

        result = train_noise_augmented_controller(
            controller=ctrl,
            env=env,
            train_episodes=2,
            max_steps=20,
            seed=43,
            checkpoint_path=ckpt_path,
            verbose=False,
        )

        assert os.path.exists(ckpt_path)
        assert result["episodes"] == 2

