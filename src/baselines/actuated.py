"""
Actuated (SCATS-style) Baseline Controller (B2)

Non-RL baseline controller dynamically extending green time based on real-time detector queue measurements.
"""

from typing import Dict
import numpy as np


class ActuatedController:
    """
    Actuated / SCATS-Style Traffic Controller (Baseline B2).
    Dynamically extends green phase if queue is detected on active approach,
    otherwise switches to opposing approach phase once min green threshold is met.
    """

    def __init__(
        self,
        min_green: int = 5,
        max_green: int = 50,
        yellow_duration: int = 3,
        queue_threshold: float = 1.0,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration
        self.queue_threshold = queue_threshold

        self.time_in_phase: Dict[str, int] = {}
        self.current_phase: Dict[str, int] = {}

    def reset(self, agent_ids: list):
        for aid in agent_ids:
            self.time_in_phase[aid] = 0
            self.current_phase[aid] = 0

    def select_action(self, agent_id: str, obs: np.ndarray) -> int:
        cur_phase = self.current_phase.get(agent_id, 0)
        t_phase = self.time_in_phase.get(agent_id, 0)

        # Parse queues from observation vector (queues on 4 approaches: N, S, E, W)
        queues = obs[:4] if len(obs) >= 4 else np.zeros(4)
        ns_queue = queues[0] + queues[1]
        ew_queue = queues[2] + queues[3]

        next_phase = cur_phase

        if cur_phase == 0:  # N-S Green
            if t_phase >= self.min_green:
                # If N-S queue cleared or max green reached, switch to yellow
                if ns_queue < self.queue_threshold or t_phase >= self.max_green or ew_queue > ns_queue + 2.0:
                    next_phase = 1
        elif cur_phase == 1:  # N-S Yellow
            if t_phase >= self.yellow_duration:
                next_phase = 2  # Switch to E-W Green
        elif cur_phase == 2:  # E-W Green
            if t_phase >= self.min_green:
                if ew_queue < self.queue_threshold or t_phase >= self.max_green or ns_queue > ew_queue + 2.0:
                    next_phase = 3
        elif cur_phase == 3:  # E-W Yellow
            if t_phase >= self.yellow_duration:
                next_phase = 0  # Switch to N-S Green

        if next_phase != cur_phase:
            self.current_phase[agent_id] = next_phase
            self.time_in_phase[agent_id] = 0
        else:
            self.time_in_phase[agent_id] = t_phase + 1

        return next_phase

    def select_actions(self, obs_dict: Dict[str, np.ndarray]) -> Dict[str, int]:
        actions = {}
        for agent_id, obs in obs_dict.items():
            actions[agent_id] = self.select_action(agent_id, obs)
        return actions
