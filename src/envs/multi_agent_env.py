"""
PettingZoo Parallel Environment Wrapper (Phase 1 §1.2 & §1.3)

Provides standard PettingZoo ParallelEnv API compatibility for MARL algorithms.
"""

from typing import Any, Dict, List, Optional, Tuple
from pettingzoo import ParallelEnv
from gymnasium import spaces
import numpy as np

from src.envs.sumo_env import SUMOEnv


class MARLTSCParallelEnv(ParallelEnv):
    """
    PettingZoo Parallel Environment wrapper for multi-intersection TSC.
    """

    metadata = {"name": "ua_hcmarl_tsc_v1", "render_modes": ["human"]}

    def __init__(
        self,
        net_file: str = "networks/grid4x4/grid4x4.net.xml",
        route_file: str = "networks/grid4x4/demand_uniform.rou.xml",
        num_intersections: int = 16,
        demand_profile: str = "uniform",
        noise_scenario: str = "clean",
        max_steps: int = 3600,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_intersections = num_intersections
        self.possible_agents = [f"intersection_{i}" for i in range(num_intersections)]
        self.agents = list(self.possible_agents)

        self.sumo_env = SUMOEnv(
            net_file=net_file,
            route_file=route_file,
            num_intersections=num_intersections,
            demand_profile=demand_profile,
            noise_scenario=noise_scenario,
            max_steps=max_steps,
            seed=seed,
        )

        self._observation_spaces = {
            agent: self.sumo_env.observation_space for agent in self.possible_agents
        }
        self._action_spaces = {
            agent: self.sumo_env.action_space for agent in self.possible_agents
        }

    def observation_space(self, agent: str) -> spaces.Space:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        return self._action_spaces[agent]

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        self.agents = list(self.possible_agents)
        obs, infos = self.sumo_env.reset(seed=seed, options=options)
        return obs, infos

    def step(
        self, actions: Dict[str, int]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Any],
    ]:
        obs, rewards, dones, truncateds, infos = self.sumo_env.step(actions)

        # Check if all agents are done
        if all(dones.values()):
            self.agents = []

        return obs, rewards, dones, truncateds, infos
