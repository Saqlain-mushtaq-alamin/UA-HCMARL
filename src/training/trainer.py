"""
Multi-Timescale Training Loop & Evaluation Orchestrator (Phase 1 / Phase 2 §2.1-2.4)

Handles training loop orchestration across baselines:
- Stage 0: Non-RL baselines (FixedTime, Actuated)
- Stage 1: Flat Unconstrained MARL (IPPO, MAPPO)
- Stage 2: Flat Constrained MARL (MAPPO-LCE)
- Stage 2b: Hierarchical MARL (FMA2C)
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np

from src.envs.sumo_env import SUMOEnv
from src.evaluation.metrics import MetricsCalculator
from src.training.callbacks import ExperimentLogger
from src.training.curriculum import CurriculumScheduler


class Trainer:
    """
    Multi-timescale training & evaluation orchestrator.
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
                    actions = controller.select_actions(obs_dict, evaluate=True, step_count=step)
                except TypeError:
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

    def train_episode(
        self, controller, max_steps: int = 1000, seed: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Runs a training episode for an RL controller, storing transitions and running updates.
        """
        self.metrics_calc.reset()
        obs_dict, _ = self.env.reset(seed=seed)

        if hasattr(controller, "reset"):
            controller.reset(list(obs_dict.keys()))

        for step in range(max_steps):
            try:
                actions = controller.select_actions(obs_dict, evaluate=False, step_count=step)
            except TypeError:
                actions = controller.select_actions(obs_dict)

            obs_dict, rewards, dones, truncateds, info = self.env.step(actions)

            # Store transition in controller buffer if supported
            if hasattr(controller, "store_transition"):
                try:
                    controller.store_transition(
                        rewards=rewards,
                        dones=dones,
                        step_violations=info.get("step_violations"),
                    )
                except TypeError:
                    controller.store_transition(rewards=rewards, dones=dones)

            self.metrics_calc.record_step(
                intersection_delays=info["step_delays"],
                intersection_queues=info["step_queues"],
                step_throughput=info["step_throughput"],
                step_violations=info["step_violations"],
                step_violation_magnitude=info["step_violation_magnitude"],
            )

            if all(dones.values()):
                break

        loss_info = {}
        if hasattr(controller, "update"):
            loss_info = controller.update()

        summary = self.metrics_calc.compute_summary()
        return summary, loss_info
