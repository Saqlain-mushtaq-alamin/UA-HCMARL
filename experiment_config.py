"""
experiment_config.py — Centralized Configuration Management for UA-HCMARL Benchmarks
=====================================================================================
Defines standardized configuration profiles ('canonical', 'fast', 'smoke')
and codifies the centralized seed allocation policy to prevent train/test/eval contamination.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Set


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
    bse_train_seed: int = 80042
    bse_test_seed: int = 81042
    training_seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    eval_base_seeds: List[int] = field(
        default_factory=lambda: [1000 + i * 100 for i in range(10)]
    )
    noise_scenarios: List[str] = field(
        default_factory=lambda: ["clean", "mild", "moderate", "severe", "extreme"]
    )
    noise_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5]
    )

    def get_eval_episode_seeds(self) -> List[int]:
        """Returns the full list of canonical evaluation episode seeds."""
        seeds = []
        for base in self.eval_base_seeds[: self.num_seeds]:
            for ep in range(self.eval_episodes_per_seed):
                seeds.append(base + ep)
        return seeds

    def verify_seed_disjointness(self) -> bool:
        """
        Validates that BSE training seeds, BSE test seeds, training replicate seeds,
        and evaluation episode seeds are strictly mutually disjoint.
        """
        bse_train_set = {self.bse_train_seed}
        bse_test_set = {self.bse_test_seed}
        train_rep_set = set(self.training_seeds)
        eval_set = set(self.get_eval_episode_seeds())

        # Check for pairwise collisions
        assert bse_train_set.isdisjoint(bse_test_set), "BSE train and test seeds overlap!"
        assert bse_train_set.isdisjoint(eval_set), "BSE train seed overlaps with evaluation seeds!"
        assert bse_test_set.isdisjoint(eval_set), "BSE test seed overlaps with evaluation seeds!"
        assert train_rep_set.isdisjoint(eval_set), "Training seeds overlap with evaluation seeds!"
        return True

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
            "bse_train_seed": self.bse_train_seed,
            "bse_test_seed": self.bse_test_seed,
            "training_seeds": self.training_seeds,
            "eval_base_seeds": self.eval_base_seeds,
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
        bse_train_seed=80042,
        bse_test_seed=81042,
        training_seeds=[42, 43, 44],
        eval_base_seeds=[1000 + i * 100 for i in range(10)],
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
        bse_train_seed=80042,
        bse_test_seed=81042,
        training_seeds=[42, 43, 44],
        eval_base_seeds=[1000 + i * 100 for i in range(5)],
    ),
    "smoke": ExperimentConfig(
        profile_name="smoke",
        num_intersections=4,
        num_seeds=2,
        max_steps=50,
        train_episodes=2,
        train_curriculum_episodes=1,
        eval_episodes_per_seed=1,
        bse_steps=100,
        bse_epochs=2,
        bse_train_seed=80042,
        bse_test_seed=81042,
        training_seeds=[42],
        eval_base_seeds=[1000, 1100],
    ),
}


def get_config(profile: str = "canonical") -> ExperimentConfig:
    """Retrieve an experiment configuration by profile name."""
    profile_lower = profile.lower().strip()
    if profile_lower not in PROFILES:
        valid = list(PROFILES.keys())
        raise ValueError(f"Unknown config profile '{profile}'. Valid profiles are: {valid}")
    cfg = PROFILES[profile_lower]
    cfg.verify_seed_disjointness()
    return cfg
