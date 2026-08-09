"""
Fixed-Time Baseline Controller (B1)

Non-RL baseline controller operating with fixed signal cycle lengths.
Default cycle: Phase 0 (30s) -> Phase 1 (3s yellow) -> Phase 2 (30s) -> Phase 3 (3s yellow)
"""

from typing import Dict, Any
import numpy as np


class FixedTimeController:
    """
    Fixed-Time Traffic Light Controller (Baseline B1).
    """

    def __init__(self, green_duration: int = 30, yellow_duration: int = 3):
        self.green_duration = green_duration
        self.yellow_duration = yellow_duration
        self.cycle_length = 2 * (green_duration + yellow_duration)

    def select_action(self, obs: np.ndarray, step_count: int) -> int:
        """
        Determines phase selection based on current step_count in cycle.
        """
        cycle_pos = step_count % self.cycle_length

        if cycle_pos < self.green_duration:
            return 0  # N-S Green
        elif cycle_pos < self.green_duration + self.yellow_duration:
            return 1  # N-S Yellow
        elif cycle_pos < 2 * self.green_duration + self.yellow_duration:
            return 2  # E-W Green
        else:
            return 3  # E-W Yellow

    def select_actions(self, obs_dict: Dict[str, np.ndarray], step_count: int = 0) -> Dict[str, int]:
        actions = {}
        for agent_id, obs in obs_dict.items():
            actions[agent_id] = self.select_action(obs, step_count)
        return actions
