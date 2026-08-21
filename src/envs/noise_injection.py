"""
Noise and Degradation Protocol Module (Novelty N5)

Standardized observation degradation wrapper for TSC domain.
Implements D1 (Gaussian Noise), D2 (Random Dropout), and D3 (Systematic Bias)
across Clean, Mild, Moderate, Severe, and Extreme scenario presets.

FIX (2026-08-21): The original implementation seeded the RNG once at construction
time and only updated it through set_seed().  When the evaluation loop created a
new SUMOEnv per (noise_scenario, seed) pair the NoiseInjector inside it was always
constructed with seed=seed, but the RNG was never mutated for different noise
levels – causing seeds 2-5 to produce IDENTICAL results across noise levels.

Fix: generate a compound seed that is a function of BOTH the experiment seed AND a
hash of the scenario name so that each (seed, scenario) pair gets a distinct RNG.
Additionally, the persistent D3 bias factors are regenerated whenever set_seed() or
set_scenario() is called so they reflect the new seed/scenario combination.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


SCENARIO_PRESETS: Dict[str, Dict[str, float]] = {
    "clean":    {"d1_p": 0.0, "d2_p": 0.0,  "d3_b": 0.0},
    "mild":     {"d1_p": 0.1, "d2_p": 0.05, "d3_b": 0.0},
    "moderate": {"d1_p": 0.2, "d2_p": 0.1,  "d3_b": 0.1},
    "severe":   {"d1_p": 0.3, "d2_p": 0.2,  "d3_b": 0.2},
    "extreme":  {"d1_p": 0.5, "d2_p": 0.3,  "d3_b": 0.3},
}

# Map scenario names to a stable integer offset so that
# (seed=3, scenario="moderate") != (seed=3, scenario="mild")
_SCENARIO_OFFSETS: Dict[str, int] = {
    "clean":    0,
    "mild":     100_000,
    "moderate": 200_000,
    "severe":   300_000,
    "extreme":  400_000,
}


def _make_compound_seed(base_seed: Optional[int], scenario: str) -> int:
    """Combines experiment seed with scenario offset to avoid identical RNGs."""
    offset = _SCENARIO_OFFSETS.get(scenario.lower(), 0)
    return (base_seed if base_seed is not None else 0) + offset


class NoiseInjector:
    """
    Applies Gaussian noise (D1), Random Dropout (D2), and Systematic Bias (D3)
    to observation arrays or feature dictionaries.
    """

    def __init__(
        self,
        scenario: str = "clean",
        d1_p: Optional[float] = None,
        d2_p: Optional[float] = None,
        d3_b: Optional[float] = None,
        heterogeneous: bool = False,
        num_intersections: int = 1,
        seed: Optional[int] = None,
    ):
        self._base_seed = seed
        self.scenario = scenario.lower()
        preset = SCENARIO_PRESETS.get(self.scenario, SCENARIO_PRESETS["clean"])

        self.d1_p = d1_p if d1_p is not None else preset["d1_p"]
        self.d2_p = d2_p if d2_p is not None else preset["d2_p"]
        self.d3_b = d3_b if d3_b is not None else preset["d3_b"]
        self.heterogeneous = heterogeneous
        self.num_intersections = num_intersections

        # Seed the RNG using the compound seed to ensure distinctness per scenario
        compound = _make_compound_seed(seed, self.scenario)
        self.rng = np.random.default_rng(compound)

        # Persistent per-intersection/per-detector bias factors (D3)
        self._bias_factors: Optional[np.ndarray] = None
        self._stale_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Setters
    # ------------------------------------------------------------------

    def set_scenario(self, scenario: str):
        """Switch degradation scenario; rebuilds RNG with compound seed."""
        self.scenario = scenario.lower()
        preset = SCENARIO_PRESETS.get(self.scenario, SCENARIO_PRESETS["clean"])
        self.d1_p = preset["d1_p"]
        self.d2_p = preset["d2_p"]
        self.d3_b = preset["d3_b"]
        # Rebuild RNG with new compound seed
        compound = _make_compound_seed(self._base_seed, self.scenario)
        self.rng = np.random.default_rng(compound)
        # Reset bias factors so they reflect the new scenario seed
        self._bias_factors = None
        self._stale_cache = {}

    def set_seed(self, seed: int):
        """Update base seed; rebuilds RNG to ensure different trajectories per seed."""
        self._base_seed = seed
        compound = _make_compound_seed(seed, self.scenario)
        self.rng = np.random.default_rng(compound)
        # Reset bias factors so they are regenerated with the new seed
        self._bias_factors = None
        self._stale_cache = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_bias(self, obs_shape: Tuple[int, ...]):
        """Initialize fixed systematic bias factors per feature dimension."""
        if self.d3_b == 0.0:
            self._bias_factors = np.ones(obs_shape, dtype=np.float32)
        else:
            # Random sign for bias per detector/feature
            signs = self.rng.choice([-1.0, 1.0], size=obs_shape)
            self._bias_factors = (1.0 + signs * self.d3_b).astype(np.float32)

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def apply_noise(
        self,
        obs: np.ndarray,
        agent_id: Optional[str] = None,
        intersection_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Applies D1, D2, and D3 noise to raw ground-truth observation array.

        Args:
            obs: Ground truth observation vector/matrix.
            agent_id: Identifier for agent caching stale values.
            intersection_idx: Index of intersection for heterogeneous scaling.

        Returns:
            Noisy observation array (clipped to non-negative).
        """
        if obs is None:
            return obs

        obs_noisy = np.copy(obs).astype(np.float32)

        # Per-intersection scaling for heterogeneous degradation
        scale = 1.0
        if self.heterogeneous and intersection_idx is not None and self.num_intersections > 1:
            scale = 0.5 + (intersection_idx / max(1, self.num_intersections - 1))

        d1_p = self.d1_p * scale
        d2_p = min(1.0, self.d2_p * scale)
        d3_b = self.d3_b * scale

        # --- D3: Systematic Bias ---
        if d3_b > 0:
            if self._bias_factors is None or self._bias_factors.shape != obs.shape:
                self._init_bias(obs.shape)
            obs_noisy = obs_noisy * self._bias_factors

        # --- D1: Gaussian Noise ---
        if d1_p > 0:
            sigma = d1_p * (np.abs(obs_noisy) + 1.0)  # +1.0 ensures noise even for zero obs
            noise = self.rng.normal(loc=0.0, scale=sigma, size=obs.shape)
            obs_noisy = obs_noisy + noise

        # Ensure traffic values remain physical (non-negative)
        obs_noisy = np.maximum(0.0, obs_noisy)

        # --- D2: Random Dropout ---
        if d2_p > 0:
            dropout_mask = self.rng.random(size=obs.shape) < d2_p
            cache_key = agent_id if agent_id is not None else "default"

            # Use stale value or 0.0 if no stale value cached yet
            stale_val = self._stale_cache.get(cache_key, np.zeros_like(obs_noisy))
            if stale_val.shape == obs_noisy.shape:
                obs_noisy[dropout_mask] = stale_val[dropout_mask]
            else:
                obs_noisy[dropout_mask] = 0.0

        # Update stale cache AFTER applying dropout (cache post-dropout value)
        if agent_id is not None:
            self._stale_cache[agent_id] = np.copy(obs_noisy)

        return obs_noisy.astype(np.float32)

    def get_config(self) -> Dict[str, Union[str, float, bool]]:
        return {
            "scenario": self.scenario,
            "d1_p": self.d1_p,
            "d2_p": self.d2_p,
            "d3_b": self.d3_b,
            "heterogeneous": self.heterogeneous,
            "num_intersections": self.num_intersections,
        }
