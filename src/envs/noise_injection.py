"""
Noise and Degradation Protocol Module (Novelty N5)

Standardized observation degradation wrapper for TSC domain.
Implements D1 (Gaussian Noise), D2 (Random Dropout), and D3 (Systematic Bias)
across Clean, Mild, Moderate, Severe, and Extreme scenario presets.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


SCENARIO_PRESETS: Dict[str, Dict[str, float]] = {
    "clean": {"d1_p": 0.0, "d2_p": 0.0, "d3_b": 0.0},
    "mild": {"d1_p": 0.1, "d2_p": 0.05, "d3_b": 0.0},
    "moderate": {"d1_p": 0.2, "d2_p": 0.1, "d3_b": 0.1},
    "severe": {"d1_p": 0.3, "d2_p": 0.2, "d3_b": 0.2},
    "extreme": {"d1_p": 0.5, "d2_p": 0.3, "d3_b": 0.3},
}


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
        self.scenario = scenario.lower()
        preset = SCENARIO_PRESETS.get(self.scenario, SCENARIO_PRESETS["clean"])

        self.d1_p = d1_p if d1_p is not None else preset["d1_p"]
        self.d2_p = d2_p if d2_p is not None else preset["d2_p"]
        self.d3_b = d3_b if d3_b is not None else preset["d3_b"]
        self.heterogeneous = heterogeneous
        self.num_intersections = num_intersections

        self.rng = np.random.default_rng(seed)

        # Persistent per-intersection/per-detector bias factors (D3)
        self._bias_factors: Optional[np.ndarray] = None
        self._stale_cache: Dict[str, np.ndarray] = {}

    def set_seed(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def _init_bias(self, obs_shape: Tuple[int, ...]):
        """Initialize fixed systematic bias factors per feature dimension."""
        if self.d3_b == 0.0:
            self._bias_factors = np.ones(obs_shape, dtype=np.float32)
        else:
            # Random sign for bias per detector/feature
            signs = self.rng.choice([-1.0, 1.0], size=obs_shape)
            self._bias_factors = (1.0 + signs * self.d3_b).astype(np.float32)

    def apply_noise(
        self,
        obs: np.ndarray,
        agent_id: Optional[str] = None,
        intersection_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Applies D1, D2, and D3 noise to raw ground-truth observation array.
        
        Args:
            obs: Ground truth observation vector/matrix (non-negative traffic counts/queues/delays).
            agent_id: Identifier for agent caching stale values.
            intersection_idx: Index of intersection for heterogeneous parameter scaling.

        Returns:
            Noisy observation array (clipped to non-negative where applicable).
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
            sigma = d1_p * np.abs(obs_noisy) + 1e-4
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
            obs_noisy[dropout_mask] = stale_val[dropout_mask] if stale_val.shape == obs_noisy.shape else 0.0

        # Update stale cache
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
