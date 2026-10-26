"""
Multi-Timescale Training Loop & Evaluation Orchestrator (Phase 1 / Phase 2 §2.1-2.4)

Handles training loop orchestration across baselines:
- Stage 0: Non-RL baselines (FixedTime, Actuated)
- Stage 1: Flat Unconstrained MARL (IPPO, MAPPO)
- Stage 2: Flat Constrained MARL (MAPPO-LCE)
- Stage 2b: Hierarchical MARL (FMA2C)

FIX (2026-08-26) -- Bug 4.2:
  Added LearningCurveLogger class and Trainer.get_learning_curves() /
  save_learning_curves() to capture per-episode metrics (delay, violations,
  policy_loss, value_loss) during training.  Without this, reviewers cannot
  verify convergence or training stability.
"""

import json
import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

from src.envs.sumo_env import SUMOEnv
from src.evaluation.metrics import MetricsCalculator
from src.training.callbacks import ExperimentLogger
from src.training.curriculum import CurriculumScheduler


class LearningCurveLogger:
    """
    Bug 4.2 fix: per-episode metric recorder.

    Records avg_delay, violation_rate, policy_loss, value_loss, and entropy_loss
    for each training episode so that learning curves can be plotted with error bands.

    Usage
    -----
    logger = LearningCurveLogger()
    trainer = Trainer(env, learning_curve_logger=logger)
    for ep in range(num_episodes):
        summary, loss = trainer.train_episode(ctrl, seed=ep)
    curves = logger.get()
    logger.save("experiments/curves_mappo.json")
    """

    def __init__(self):
        self._curves: Dict[str, List[float]] = {
            "episode": [],
            "avg_delay": [],
            "violation_rate": [],
            "policy_loss": [],
            "value_loss": [],
            "entropy_loss": [],
            "worker_loss": [],
            "manager_loss": [],
        }

    def record(self, episode: int, summary: Dict[str, float], loss_info: Dict[str, float]):
        self._curves["episode"].append(episode)
        self._curves["avg_delay"].append(float(summary.get("avg_delay", float("nan"))))
        self._curves["violation_rate"].append(
            float(summary.get("constraint_violation_rate", float("nan")))
        )
        self._curves["policy_loss"].append(float(loss_info.get("policy_loss", float("nan"))))
        self._curves["value_loss"].append(float(loss_info.get("value_loss", float("nan"))))
        self._curves["entropy_loss"].append(float(loss_info.get("entropy_loss", float("nan"))))
        self._curves["worker_loss"].append(float(loss_info.get("worker_loss", float("nan"))))
        self._curves["manager_loss"].append(float(loss_info.get("manager_loss", float("nan"))))

    def get(self) -> Dict[str, List[float]]:
        return dict(self._curves)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._curves, f, indent=2)

    def reset(self):
        for key in self._curves:
            self._curves[key] = []


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
        learning_curve_logger: Optional[LearningCurveLogger] = None,
    ):
        self.env = env
        self.logger = logger
        self.curriculum = curriculum or CurriculumScheduler()
        self.t_manager = t_manager
        self.metrics_calc = MetricsCalculator(num_intersections=env.num_intersections)
        # Bug 4.2: optional learning-curve logger attached at construction time
        self.learning_curve_logger = learning_curve_logger
        self._episode_count = 0

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
                step_violations_per_agent=info.get("step_violations_per_agent"),
            )

            if all(dones.values()):
                break

        return self.metrics_calc.compute_summary()

    def train_episode(
        self, controller, max_steps: int = 1000, seed: Optional[int] = None
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Runs a training episode for an RL controller, storing transitions and running updates.

        If a LearningCurveLogger is attached, records per-episode metrics automatically.
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
                step_violations_per_agent=info.get("step_violations_per_agent"),
            )

            if all(dones.values()):
                break

        loss_info = {}
        if hasattr(controller, "update"):
            loss_info = controller.update()

        summary = self.metrics_calc.compute_summary()

        # Bug 4.2: record into attached learning-curve logger if present
        if self.learning_curve_logger is not None:
            self.learning_curve_logger.record(self._episode_count, summary, loss_info)
        self._episode_count += 1

        return summary, loss_info

    def get_learning_curves(self) -> Optional[Dict[str, List[float]]]:
        """Returns learning curve data if a LearningCurveLogger is attached."""
        if self.learning_curve_logger is not None:
            return self.learning_curve_logger.get()
        return None

    def save_learning_curves(self, path: str):
        """Saves learning curve data to JSON. No-op if no logger attached."""
        if self.learning_curve_logger is not None:
            self.learning_curve_logger.save(path)
