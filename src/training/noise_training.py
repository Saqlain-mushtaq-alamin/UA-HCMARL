"""
Noise-Trained Baseline Training Module (Phase 3 Blocker 2 Resolution)
=====================================================================
Implements multi-noise curriculum training for IPPO and FMA2C (Arms M2 and M5).

Key Features:
- Observation noise is injected dynamically during training, exposing policies to the
  full spectrum of noise scenarios (Clean, Mild, Moderate, Severe, Extreme; sigma in [0, 0.5]).
- Domain randomization across presets: scenario is sampled per episode using a dedicated,
  reproducible random stream that does not perturb traffic generation or evaluation seeds.
- Matches training budget (episodes, steps, learning rates) to Phase 2 clean baselines.
- Saves model checkpoints into Phase 3 canonical storage, leaving Phase 2 artifacts untouched.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from src.envs.sumo_env import SUMOEnv
from src.training.trainer import Trainer, LearningCurveLogger


CANONICAL_NOISE_SCENARIOS = ["clean", "mild", "moderate", "severe", "extreme"]


def train_noise_augmented_controller(
    controller: Any,
    env: SUMOEnv,
    train_episodes: int = 200,
    max_steps: int = 3600,
    seed: int = 42,
    noise_scenarios: Optional[List[str]] = None,
    checkpoint_path: Optional[str] = None,
    log_curve_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trains a controller under a multi-noise observation mixture.

    Parameters
    ----------
    controller : Any
        Controller instance (IPPOController, FMA2CController).
    env : SUMOEnv
        Training SUMO environment.
    train_episodes : int
        Total training episodes.
    max_steps : int
        Simulation horizon per episode.
    seed : int
        Base replicate seed (e.g. 42, 43, 44).
    noise_scenarios : List[str]
        Noise scenarios to sample from during training.
    checkpoint_path : Optional[str]
        File path to save the final trained checkpoint.
    log_curve_path : Optional[str]
        File path to save the JSON learning curve.
    verbose : bool
        Whether to print periodic progress.

    Returns
    -------
    Dict[str, Any]
        Training summary with episode history and sampled scenarios.
    """
    if noise_scenarios is None:
        noise_scenarios = list(CANONICAL_NOISE_SCENARIOS)

    # Dedicated noise sampling RNG, decoupled from environment traffic seed
    noise_rng = np.random.default_rng(seed * 10007 + 777)

    logger = LearningCurveLogger()
    trainer = Trainer(env=env, learning_curve_logger=logger)

    episode_scenarios = []
    episode_delays = []
    episode_violations = []

    if verbose:
        ctrl_name = type(controller).__name__
        print(f"  [Noise-Training] Starting {ctrl_name} (seed={seed}, eps={train_episodes}, steps={max_steps})...")

    for ep in range(train_episodes):
        # 1. Deterministically sample noise scenario for this episode
        scenario = str(noise_rng.choice(noise_scenarios))
        episode_scenarios.append(scenario)

        # 2. Inject noise scenario into environment
        env.noise_injector.set_scenario(scenario)

        # 3. Train episode with unique episode seed
        ep_seed = seed * 1000 + ep
        summary, loss_info = trainer.train_episode(
            controller, max_steps=max_steps, seed=ep_seed
        )

        avg_delay = float(summary.get("avg_delay", 0.0))
        viol_rate = float(summary.get("constraint_violation_rate", 0.0))
        episode_delays.append(avg_delay)
        episode_violations.append(viol_rate)

        if verbose and ((ep + 1) % max(1, train_episodes // 10) == 0 or ep == train_episodes - 1):
            recent_delay = np.mean(episode_delays[-10:])
            print(f"    Ep {ep+1:3d}/{train_episodes:3d} [{scenario.upper():8s}] | Delay: {recent_delay:5.2f} veh | Viol: {viol_rate*100.0:4.1f}%")

    # 4. Save checkpoint if requested
    if checkpoint_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
        if hasattr(controller, "save_model"):
            controller.save_model(checkpoint_path)
            if verbose:
                print(f"  [Noise-Training] Checkpoint saved -> {checkpoint_path}")

    # 5. Save learning curves
    if log_curve_path is not None:
        logger.save(log_curve_path)

    return {
        "seed": seed,
        "episodes": train_episodes,
        "scenarios_sampled": episode_scenarios,
        "final_delay": float(np.mean(episode_delays[-10:])),
        "final_violation_rate": float(np.mean(episode_violations[-10:])),
        "checkpoint_path": checkpoint_path,
    }

