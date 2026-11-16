"""
Noise and Degradation Protocol V2 (Detector-Only Degradation)
============================================================
Audited and separated from V1. Corrupts ONLY detector-derived observation
dimensions, preserving internal controller clock, signal phases, and
static configuration parameters.

Dimension Map:
- Dims [0, 1, 2, 3]: Approach queues (Detector-derived) -> DEGRADED
- Dims [4, 5, 6, 7]: Arrival rates (Config constant)    -> CLEAN
- Dims [8, 9, 10, 11]: Phase one-hot (Controller state) -> CLEAN
- Dim 12: Time in phase (Controller clock)              -> CLEAN
- Dims [13, 14, 15]: Total queue, pressure, throughput  -> DEGRADED
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from src.envs.noise_injection import SCENARIO_PRESETS, _make_compound_seed

DETECTOR_DIMS = [0, 1, 2, 3, 13, 14, 15]
CONTROLLER_DIMS = [4, 5, 6, 7, 8, 9, 10, 11, 12]

class NoiseInjectorV2:
    """
    Applies Gaussian noise (D1), Random Dropout (D2), and Systematic Bias (D3)
    exclusively to detector-derived observation dimensions.
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

        compound = _make_compound_seed(seed, self.scenario)
        self.rng = np.random.default_rng(compound)

        self._bias_factors: Optional[np.ndarray] = None
        self._stale_cache: Dict[str, np.ndarray] = {}

    def set_scenario(self, scenario: str):
        self.scenario = scenario.lower()
        preset = SCENARIO_PRESETS.get(self.scenario, SCENARIO_PRESETS["clean"])
        self.d1_p = preset["d1_p"]
        self.d2_p = preset["d2_p"]
        self.d3_b = preset["d3_b"]
        compound = _make_compound_seed(self._base_seed, self.scenario)
        self.rng = np.random.default_rng(compound)
        self._bias_factors = None
        self._stale_cache = {}

    def set_seed(self, seed: int):
        self._base_seed = seed
        compound = _make_compound_seed(seed, self.scenario)
        self.rng = np.random.default_rng(compound)
        self._bias_factors = None
        self._stale_cache = {}

    def _init_bias(self, num_detector_dims: int):
        if self.d3_b == 0.0:
            self._bias_factors = np.ones(num_detector_dims, dtype=np.float32)
        else:
            signs = self.rng.choice([-1.0, 1.0], size=num_detector_dims)
            self._bias_factors = (1.0 + signs * self.d3_b).astype(np.float32)

    def apply_noise(
        self,
        obs: np.ndarray,
        agent_id: Optional[str] = None,
        intersection_idx: Optional[int] = None,
    ) -> np.ndarray:
        if obs is None:
            return obs

        obs_noisy = np.copy(obs).astype(np.float32)
        det_indices = DETECTOR_DIMS
        det_vals = obs_noisy[det_indices]

        scale = 1.0
        if self.heterogeneous and intersection_idx is not None and self.num_intersections > 1:
            scale = 0.5 + (intersection_idx / max(1, self.num_intersections - 1))

        d1_p = self.d1_p * scale
        d2_p = min(1.0, self.d2_p * scale)
        d3_b = self.d3_b * scale

        # --- D3: Systematic Bias (Detector Dims Only) ---
        if d3_b > 0:
            if self._bias_factors is None or self._bias_factors.shape[0] != len(det_indices):
                self._init_bias(len(det_indices))
            det_vals = det_vals * self._bias_factors

        # --- D1: Gaussian Noise (Detector Dims Only) ---
        if d1_p > 0:
            sigma = d1_p * (np.abs(det_vals) + 1.0)
            noise = self.rng.normal(loc=0.0, scale=sigma, size=det_vals.shape)
            det_vals = det_vals + noise

        det_vals = np.maximum(0.0, det_vals)

        # --- D2: Random Dropout (Detector Dims Only) ---
        if d2_p > 0:
            dropout_mask = self.rng.random(size=det_vals.shape) < d2_p
            cache_key = agent_id if agent_id is not None else "default"
            stale_val = self._stale_cache.get(cache_key, np.zeros_like(det_vals))
            if stale_val.shape == det_vals.shape:
                det_vals[dropout_mask] = stale_val[dropout_mask]
            else:
                det_vals[dropout_mask] = 0.0

        if agent_id is not None:
            self._stale_cache[agent_id] = np.copy(det_vals)

        obs_noisy[det_indices] = det_vals
        # Controller internal dims (4-12) are completely preserved as clean
        return obs_noisy.astype(np.float32)

    def get_config(self) -> Dict[str, Union[str, float, bool]]:
        return {
            "scenario": self.scenario,
            "version": "v2_detector_only",
            "d1_p": self.d1_p,
            "d2_p": self.d2_p,
            "d3_b": self.d3_b,
            "heterogeneous": self.heterogeneous,
            "num_intersections": self.num_intersections,
        }

