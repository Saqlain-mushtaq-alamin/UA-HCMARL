"""
Automated Seed Allocation & Trajectory Disjointness Tests (Phase 3 Blocker 3)
=============================================================================
Proves mathematical disjointness of seed spaces and confirms distinct physical
simulation trajectories between BSE training, validation, and evaluation streams.
"""

import pytest
import numpy as np
from experiment_config import get_config, PROFILES
from src.envs.sumo_env import SUMOEnv


def test_seed_set_disjointness_across_all_profiles():
    """Verify that all profiles satisfy strict seed set disjointness."""
    for profile_name in ["canonical", "fast", "smoke"]:
        cfg = get_config(profile_name)
        assert cfg.verify_seed_disjointness() is True

        eval_seeds = set(cfg.get_eval_episode_seeds())
        assert 80042 not in eval_seeds, "BSE train seed 80042 collides with eval seeds!"
        assert 81042 not in eval_seeds, "BSE test seed 81042 collides with eval seeds!"
        assert 1042 not in eval_seeds, "Old contaminated seed 1042 must not be in eval seeds!"


def test_trajectory_level_disjointness():
    """
    Verify at the physical simulator queue level that BSE train seed (80042),
    BSE test seed (81042), and canonical evaluation seeds produce distinct trajectories.
    """
    def run_trajectory(seed: int, steps: int = 20) -> np.ndarray:
        env = SUMOEnv(num_intersections=4, max_steps=steps, seed=seed)
        obs, _ = env.reset(seed=seed)
        queues = []
        for _ in range(steps):
            actions = {aid: 0 for aid in env.intersection_ids}
            obs, rewards, _, _, info = env.step(actions)
            queues.append(list(info["step_queues"].values()))
        return np.array(queues)

    traj_bse_train = run_trajectory(80042)
    traj_bse_test  = run_trajectory(81042)
    traj_eval_1000 = run_trajectory(1000)
    traj_eval_1001 = run_trajectory(1001)

    # 1. Train and test BSE data must be physically distinct
    diff_train_test = np.abs(traj_bse_train - traj_bse_test).sum()
    assert diff_train_test > 1.0, f"BSE train and test trajectories identical! diff={diff_train_test}"

    # 2. BSE test data must be physically distinct from evaluation stream
    diff_test_eval = np.abs(traj_bse_test - traj_eval_1000).sum()
    assert diff_test_eval > 1.0, f"BSE test and eval trajectories identical! diff={diff_test_eval}"

    # 3. Consecutive evaluation seeds must be distinct
    diff_evals = np.abs(traj_eval_1000 - traj_eval_1001).sum()
    assert diff_evals > 1.0, f"Evaluation seeds 1000 and 1001 identical! diff={diff_evals}"

