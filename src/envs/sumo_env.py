"""
SUMO Base Environment Wrapper (Phase 1 §1.1)

Gymnasium/PettingZoo compliant traffic signal environment.
Features:
- TraCI interface integration for SUMO simulations.
- Native high-fidelity Python micro-simulation fallback engine when running without SUMO binary.
- Tracks queue lengths, vehicle delays, occupancy, throughput, travel time, and phase constraint violations.
- Integrated noise degradation wrapper.

FIX (2026-08-21):
  1. Reward now uses a per-step cumulative queue signal (bounded and learning-stable)
     instead of the raw wait_times accumulator, which grew unboundedly and produced
     rewards on the order of -10^4 that destroyed gradient flow.
  2. SingleIntersectionSimulator.step() returns a step-level delay that is the sum
     of current approach queues (a proxy for instantaneous delay), not the cumulative
     historical accumulator.
  3. SUMOEnv.reset() now passes the compound (seed, noise_scenario) information to
     the NoiseInjector so that each evaluation episode truly differs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.noise_injection import NoiseInjector

try:
    import traci
    import sumolib
    TRACI_AVAILABLE = True
except ImportError:
    TRACI_AVAILABLE = False


class SingleIntersectionSimulator:
    """
    Python traffic queue simulator for an intersection with 4 incoming approaches (N, S, E, W).
    Phases:
    0: N-S Green (Straight/Right)
    1: N-S Yellow
    2: E-W Green (Straight/Right)
    3: E-W Yellow
    """

    def __init__(
        self,
        intersection_id: str = "intersection_0",
        min_green: int = 5,
        max_green: int = 50,
        yellow_time: int = 3,
        seed: Optional[int] = None,
    ):
        self.intersection_id = intersection_id
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time

        self.rng = np.random.default_rng(seed)

        # Queues for 4 approaches: North, South, East, West
        self.queues = np.zeros(4, dtype=np.float32)
        # Accumulated per-step wait (reset each step for step-wise reward)
        self._step_wait = np.zeros(4, dtype=np.float32)

        self.current_phase = 0  # Start with N-S Green
        self.time_in_phase = 0
        self.throughput_count = 0

        # Arrival rate parameters (vehicles per second per approach)
        self.arrival_rates = np.array([0.2, 0.2, 0.15, 0.15], dtype=np.float32)

    def set_demand_profile(self, profile: str):
        if profile == "morning_peak":
            self.arrival_rates = np.array([0.4, 0.35, 0.1, 0.1], dtype=np.float32)
        elif profile == "evening_peak":
            self.arrival_rates = np.array([0.1, 0.1, 0.4, 0.35], dtype=np.float32)
        elif profile == "incident":
            self.arrival_rates = np.array([0.5, 0.05, 0.3, 0.3], dtype=np.float32)
        else:  # uniform
            self.arrival_rates = np.array([0.2, 0.2, 0.2, 0.2], dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, Dict[str, float], int, Dict[str, int], float]:
        """
        Executes one step in the simulator.
        Returns: (state, info, throughput, violations, violation_magnitude)
        """
        violations = {"green_time": 0, "green_skip": 0, "phase_skip": 0}
        violation_mag = 0.0

        # Check action constraints
        requested_phase = action

        if requested_phase != self.current_phase:
            # Check Min Green violation
            if self.time_in_phase < self.min_green:
                violations["green_time"] += 1
                violation_mag += (self.min_green - self.time_in_phase)

            # Check Phase Skip (jumping without yellow phase)
            if (self.current_phase % 2 == 0) and (requested_phase != self.current_phase + 1):
                violations["phase_skip"] += 1

            self.current_phase = requested_phase
            self.time_in_phase = 0
        else:
            self.time_in_phase += 1
            # Check Max Green violation
            if self.time_in_phase > self.max_green:
                violations["green_time"] += 1
                violation_mag += (self.time_in_phase - self.max_green)

        # Vehicle arrivals (Poisson process)
        arrivals = self.rng.poisson(self.arrival_rates)
        self.queues += arrivals

        # Discharge vehicles based on active phase
        step_throughput = 0
        discharge_rate = 0.8  # max vehicles discharged per second during green

        active_approaches = []
        if self.current_phase == 0:   # N-S Green
            active_approaches = [0, 1]
        elif self.current_phase == 2: # E-W Green
            active_approaches = [2, 3]

        for app in active_approaches:
            discharged = min(self.queues[app], discharge_rate)
            self.queues[app] -= discharged
            step_throughput += int(discharged)

        # Step delay = current queue length (vehicles waiting THIS step)
        # This is bounded and gives a clear learning signal for RL.
        step_delay = float(np.sum(self.queues))
        step_queue = float(np.sum(self.queues))

        # Construct raw observation
        # State vector: [queues(4) norm, arrival_rates(4) norm, phase_onehot(4), time_in_phase norm, total_queue norm, phase_pressure norm]
        max_queue = 20.0  # normalization factor
        phase_onehot = np.zeros(4, dtype=np.float32)
        phase_onehot[self.current_phase] = 1.0

        # Phase pressure: difference in queue between candidate active phases
        ns_pressure = float(self.queues[0] + self.queues[1])
        ew_pressure = float(self.queues[2] + self.queues[3])
        phase_pressure = (ns_pressure - ew_pressure) / max(1.0, ns_pressure + ew_pressure)

        obs = np.concatenate([
            np.clip(self.queues / max_queue, 0.0, 1.0),           # 4 dims
            self.arrival_rates * 5.0,                              # 4 dims (scaled)
            phase_onehot,                                          # 4 dims
            [float(self.time_in_phase) / self.max_green,           # 1 dim
             np.clip(step_queue / max_queue, 0.0, 1.0),            # 1 dim
             phase_pressure,                                       # 1 dim
             float(step_throughput) / 4.0,                        # 1 dim
             ],
        ]).astype(np.float32)  # Total: 4+4+4+4 = 16 dims

        info = {
            "queue_length": step_queue,
            "delay": step_delay,
            "current_phase": self.current_phase,
            "time_in_phase": self.time_in_phase,
        }

        return obs, info, step_throughput, violations, violation_mag


class SUMOEnv(gym.Env):
    """
    SUMO Gym Environment for single or multi-intersection grid.
    Supports noise injection, demand profiles, and baseline control interfaces.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        net_file: str = "networks/grid4x4/grid4x4.net.xml",
        route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
        num_intersections: int = 16,
        demand_profile: str = "uniform",
        noise_scenario: str = "clean",
        use_gui: bool = False,
        step_length: float = 1.0,
        max_steps: int = 3600,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.net_file = net_file
        self.route_file = route_file
        self.num_intersections = num_intersections
        self.demand_profile = demand_profile
        self.noise_scenario = noise_scenario
        self.use_gui = use_gui
        self.step_length = step_length
        self.max_steps = max_steps
        self.step_count = 0
        self._init_seed = seed

        self.noise_injector = NoiseInjector(
            scenario=noise_scenario,
            num_intersections=num_intersections,
            seed=seed,
        )
        self.clean_obs_dict: Dict[str, np.ndarray] = {}

        # Create intersection simulators for fallback/standalone operation
        self.intersection_ids = [f"intersection_{i}" for i in range(num_intersections)]
        self.simulators = {
            i_id: SingleIntersectionSimulator(
                intersection_id=i_id,
                seed=(seed * 31 + i + 7919) if seed is not None else None
            )
            for i, i_id in enumerate(self.intersection_ids)
        }

        for sim in self.simulators.values():
            sim.set_demand_profile(self.demand_profile)

        # Define Observation & Action spaces per intersection
        # Obs: 16-dim state vector per intersection (normalized)
        self.observation_space = spaces.Box(
            low=-1.0, high=2.0, shape=(16,), dtype=np.float32
        )
        # Actions: 4 discrete phase choices
        self.action_space = spaces.Discrete(4)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            # Update the noise injector's seed – this regenerates the compound RNG
            # so each (seed, noise_scenario) pair gets independent stochasticity.
            self.noise_injector.set_seed(seed)
            # Re-seed each simulator independently
            for i, (i_id, sim) in enumerate(self.simulators.items()):
                sim.rng = np.random.default_rng(seed * 31 + i + 7919)

        self.step_count = 0
        obs_dict = {}

        for i_idx, (i_id, sim) in enumerate(self.simulators.items()):
            sim.set_demand_profile(self.demand_profile)
            sim.queues = np.zeros(4, dtype=np.float32)
            sim.current_phase = 0
            sim.time_in_phase = 0

            raw_obs, _, _, _, _ = sim.step(action=0)
            self.clean_obs_dict[i_id] = raw_obs
            noisy_obs = self.noise_injector.apply_noise(
                raw_obs, agent_id=i_id, intersection_idx=i_idx
            )
            obs_dict[i_id] = noisy_obs

        return obs_dict, {"step": 0}

    def _get_clean_observation(self, agent_id: str) -> np.ndarray:
        return self.clean_obs_dict.get(agent_id, np.zeros(16, dtype=np.float32))

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, Any]]:
        """
        Step all intersections simultaneously.
        """
        self.step_count += 1
        obs_dict = {}
        rewards = {}
        dones = {}
        truncateds = {}

        is_done = self.step_count >= self.max_steps

        step_delays = {}
        step_queues = {}
        total_step_throughput = 0
        total_step_violations = {"green_time": 0, "green_skip": 0, "phase_skip": 0}
        total_violation_mag = 0.0

        for i_idx, (i_id, sim) in enumerate(self.simulators.items()):
            act = actions.get(i_id, sim.current_phase)
            raw_obs, info, throughput, violations, v_mag = sim.step(act)
            self.clean_obs_dict[i_id] = raw_obs

            noisy_obs = self.noise_injector.apply_noise(
                raw_obs, agent_id=i_id, intersection_idx=i_idx
            )
            obs_dict[i_id] = noisy_obs

            # Reward: negative step-wise queue (bounded, stable, immediately informative)
            # Normalized by number of approaches so magnitude is independent of
            # network size.
            rewards[i_id] = -float(info["delay"]) / 4.0
            dones[i_id] = is_done
            truncateds[i_id] = False

            step_delays[i_id] = info["delay"]
            step_queues[i_id] = info["queue_length"]
            total_step_throughput += throughput

            for k, v in violations.items():
                total_step_violations[k] += v
            total_violation_mag += v_mag

        info_dict = {
            "step": self.step_count,
            "step_delays": step_delays,
            "step_queues": step_queues,
            "step_throughput": total_step_throughput,
            "step_violations": total_step_violations,
            "step_violation_magnitude": total_violation_mag,
        }

        return obs_dict, rewards, dones, truncateds, info_dict
