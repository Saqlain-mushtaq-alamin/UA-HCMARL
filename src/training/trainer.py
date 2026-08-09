"""
Multi-Timescale Training Loop Skeleton (Phase 1 §1.3)

Handles training loop orchestration across Layer 1 (BSE), Layer 2 (Worker), Layer 3 (Manager).
"""

from typing import Dict, Any, Optional
import numpy as np

from src.envs.sumo_env import SUMOEnv
from src.evaluation.metrics import MetricsCalculator
from src.training.callbacks import ExperimentLogger
from src.training.curriculum import CurriculumScheduler


class Trainer:
    """
    Multi-timescale training orchestrator.
    """

    def __init__(
        self,
        env: SUMOEnv,
        logger: Optional[ExperimentLogger] = None,
        curriculum: Optional[CurriculumScheduler] = None,
        t_manager: int = 15,
    ):
        self.env = env
        self.logger = logger
        self.curriculum = curriculum or CurriculumScheduler()
        self.t_manager = t_manager
        self.metrics_calc = MetricsCalculator(num_intersections=env.num_intersections)

    def run_evaluation_episode(
        self, controller, max_steps: int = 1000, seed: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Runs a complete evaluation episode using given controller (baseline or RL policy).
        """
        self.metrics_calc.reset()
        obs_dict, _ = self.env.reset(seed=seed)

        for step in range(max_steps):
            if hasattr(controller, "select_actions"):
                if hasattr(controller, "reset") and step == 0:
                    controller.reset(list(obs_dict.keys()))
                try:
                    actions = controller.select_actions(obs_dict, step_count=step)
                except TypeError:
                    actions = controller.select_actions(obs_dict)
            else:
                actions = {aid: self.env.action_space.sample() for aid in obs_dict.keys()}

            obs_dict, rewards, dones, truncateds, info = self.env.step(actions)

            self.metrics_calc.record_step(
                intersection_delays=info["step_delays"],
                intersection_queues=info["step_queues"],
                step_throughput=info["step_throughput"],
                step_violations=info["step_violations"],
                step_violation_magnitude=info["step_violation_magnitude"],
            )

            if all(dones.values()):
                break

        return self.metrics_calc.compute_summary()
