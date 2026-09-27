"""
experiment_config.py — Centralized Configuration Management for UA-HCMARL Benchmarks
=====================================================================================
Defines standardized configuration profiles ('canonical', 'fast', 'smoke_test')
to eliminate inconsistent max_steps, seed counts, and network dimensions across phases.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ExperimentConfig:
    profile_name: str
    net_file: str = "networks/grid4x4/grid4x4.net.xml"
    route_file: str = "networks/grid4x4/demand_uniform.rou.xml"
    num_intersections: int = 16
    num_seeds: int = 10
    max_steps: int = 3600
    train_episodes: int = 200
    train_curriculum_episodes: int = 5
    eval_episodes_per_seed: int = 3
    bse_steps: int = 2000
    bse_epochs: int = 20
    noise_scenarios: List[str] = field(
        default_factory=lambda: ["clean", "mild", "moderate", "severe", "extreme"]
    )
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5]
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile_name,
            "net_file": self.net_file,
            "route_file": self.route_file,
            "num_intersections": self.num_intersections,
            "num_seeds": self.num_seeds,
            "max_steps": self.max_steps,
            "train_episodes": self.train_episodes,
            "train_curriculum_episodes": self.train_curriculum_episodes,
            "eval_episodes_per_seed": self.eval_episodes_per_seed,
            "bse_steps": self.bse_steps,
            "bse_epochs": self.bse_epochs,
            "noise_scenarios": self.noise_scenarios,
            "noise_levels": self.noise_levels,
        }


PROFILES: Dict[str, ExperimentConfig] = {
    "canonical": ExperimentConfig(
        profile_name="canonical",
        num_intersections=16,
        num_seeds=10,
        max_steps=3600,
        train_episodes=200,
        train_curriculum_episodes=5,
        eval_episodes_per_seed=3,
        bse_steps=2000,
        bse_epochs=20,
    ),
    "fast": ExperimentConfig(
        profile_name="fast",
        num_intersections=16,
        num_seeds=5,
        max_steps=500,
        train_episodes=10,
        train_curriculum_episodes=3,
        eval_episodes_per_seed=1,
        bse_steps=500,
        bse_epochs=5,
    ),
    "smoke": ExperimentConfig(
        profile_name="smoke",
        num_intersections=4,
        num_seeds=2,
        max_steps=100,
        train_episodes=2,
        train_curriculum_episodes=1,
        eval_episodes_per_seed=1,
        bse_steps=200,
        bse_epochs=2,
    ),
}


def get_config(profile: str = "canonical") -> ExperimentConfig:
    """Retrieve an experiment configuration by profile name."""
    profile_lower = profile.lower().strip()
    if profile_lower not in PROFILES:
        valid = list(PROFILES.keys())
        raise ValueError(f"Unknown config profile '{profile}'. Valid profiles are: {valid}")
    return PROFILES[profile_lower]
