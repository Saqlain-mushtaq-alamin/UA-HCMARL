"""
Max-Pressure Traffic Signal Controller (Varaiya 2013).

Classical distributed feedback controller that selects phases to maximize
movement pressure (upstream queue minus downstream queue).
In grid networks with uniform capacity, movement pressure is proportional to
the sum of queues on incoming approaches for that phase.
"""

from typing import Dict, List, Optional
import numpy as np


class MaxPressureController:
    """
    Max-Pressure Controller (Varaiya 2013).
    Evaluates queue pressure for candidate green phases (Phase 0: N-S, Phase 2: E-W)
    and switches to the phase with maximal queue pressure, subject to minimum green
    and non-preemptible yellow clearance intervals.
    """

    def __init__(
        self,
        min_green: int = 5,
        max_green: int = 50,
        yellow_duration: int = 3,
        pressure_threshold: float = 1.0,
        max_queue: float = 20.0,
    ):
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_duration = yellow_duration
        self.pressure_threshold = pressure_threshold
        self.max_queue = max_queue

        self.current_phase: Dict[str, int] = {}
        self.time_in_phase: Dict[str, int] = {}

    def reset(self, agent_ids: Optional[List[str]] = None):
        if agent_ids:
            for aid in agent_ids:
                self.current_phase[aid] = 0
                self.time_in_phase[aid] = 0

    def select_action(self, agent_id: str, obs: np.ndarray) -> int:
        """
        Determines phase choice using queue pressure.
        obs vector layout:
        obs[0:4] = queues [North, South, East, West] (normalized by max_queue)
        """
        if agent_id not in self.current_phase:
            self.current_phase[agent_id] = 0
            self.time_in_phase[agent_id] = 0

        curr_phase = self.current_phase[agent_id]
        time_in_p = self.time_in_phase[agent_id]

        # Yellow Clearance Phase: non-preemptible
        if curr_phase % 2 != 0:
            time_in_p += 1
            if time_in_p >= self.yellow_duration:
                # Transition to next green (1 -> 2 E-W, 3 -> 0 N-S)
                curr_phase = (curr_phase + 1) % 4
                time_in_p = 0
            self.current_phase[agent_id] = curr_phase
            self.time_in_phase[agent_id] = time_in_p
            return curr_phase

        # Green Phase Control (Phase 0 or Phase 2)
        # Un-normalize queues back to vehicle counts
        raw_queues = obs[0:4] * self.max_queue
        ns_pressure = float(raw_queues[0] + raw_queues[1])
        ew_pressure = float(raw_queues[2] + raw_queues[3])

        if time_in_p < self.min_green:
            # Must satisfy minimum green
            time_in_p += 1
            action = curr_phase
        elif time_in_p >= self.max_green:
            # Forced switch on max-green
            action = (curr_phase + 1) % 4
            curr_phase = action
            time_in_p = 0
        else:
            # Max-Pressure decision
            if curr_phase == 0:  # N-S Green
                if ew_pressure > ns_pressure + self.pressure_threshold:
                    # Switch to N-S Yellow clearance (Phase 1)
                    action = 1
                    curr_phase = 1
                    time_in_p = 0
                else:
                    # Hold N-S Green
                    time_in_p += 1
                    action = 0
            elif curr_phase == 2:  # E-W Green
                if ns_pressure > ew_pressure + self.pressure_threshold:
                    # Switch to E-W Yellow clearance (Phase 3)
                    action = 3
                    curr_phase = 3
                    time_in_p = 0
                else:
                    # Hold E-W Green
                    time_in_p += 1
                    action = 2
            else:
                action = curr_phase

        self.current_phase[agent_id] = curr_phase
        self.time_in_phase[agent_id] = time_in_p
        return action

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        evaluate: bool = True,
        step_count: int = 0,
        action_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, int]:
        actions = {}
        for aid, obs in obs_dict.items():
            actions[aid] = self.select_action(aid, obs)
        return actions
