"""
Unit tests for src/envs/noise_injection.py
"""

import numpy as np
import pytest
from src.envs.noise_injection import NoiseInjector, SCENARIO_PRESETS


def test_scenario_presets():
    injector = NoiseInjector(scenario="clean")
    assert injector.d1_p == 0.0
    assert injector.d2_p == 0.0
    assert injector.d3_b == 0.0

    injector_severe = NoiseInjector(scenario="severe")
    assert injector_severe.d1_p == 0.3
    assert injector_severe.d2_p == 0.2
    assert injector_severe.d3_b == 0.2


def test_d1_gaussian_noise():
    injector = NoiseInjector(scenario="clean", d1_p=0.5, seed=42)
    obs = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    noisy = injector.apply_noise(obs)
    
    assert noisy.shape == obs.shape
    assert not np.array_equal(obs, noisy)
    assert np.all(noisy >= 0.0)  # Non-negative constraint


def test_d2_random_dropout():
    injector = NoiseInjector(scenario="clean", d2_p=1.0, seed=42)  # 100% dropout
    obs = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    noisy = injector.apply_noise(obs, agent_id="agent_1")
    
    # Second step with 100% dropout should return stale value
    obs2 = np.array([15.0, 25.0, 35.0], dtype=np.float32)
    noisy2 = injector.apply_noise(obs2, agent_id="agent_1")
    assert np.array_equal(noisy2, noisy)


def test_d3_systematic_bias():
    injector = NoiseInjector(scenario="clean", d3_b=0.2, seed=42)
    obs = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    noisy = injector.apply_noise(obs)
    
    assert not np.array_equal(obs, noisy)
    # Bias factors should remain consistent across calls
    noisy2 = injector.apply_noise(obs)
    assert np.array_equal(noisy, noisy2)
