"""
Unit tests for Phase 5 Controlled Ablation Suite
"""

import pytest
import numpy as np
from src.baselines.ua_hcmarl import UAHCMARLController
from src.baselines.bse_mappo_lce import BSEMAPPOLCEController
from src.baselines.mappo import MAPPOController
from scripts.run_phase5_ablation_benchmark import generate_latex_table


def test_ablation_variants_instantiation():
    agent_ids = ["intersection_0", "intersection_1"]

    a0 = UAHCMARLController(agent_ids=agent_ids, obs_dim=16, action_dim=4)
    a1 = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=True)
    a3 = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=False)
    a6 = MAPPOController(agent_ids=agent_ids)

    obs_dict = {aid: np.random.randn(16).astype(np.float32) for aid in agent_ids}

    for name, ctrl in [("A0", a0), ("A1", a1), ("A3", a3), ("A6", a6)]:
        acts = ctrl.select_actions(obs_dict, evaluate=True)
        assert len(acts) == 2, f"Failed for variant {name}"


def test_generate_latex_table():
    results = {
        "A0: UA-HCMARL": {"avg_delay": [500.0, 510.0, 490.0], "violation_rate": [0.5, 0.52, 0.48]},
        "A5: MAPPO-LCE": {"avg_delay": [600.0, 610.0, 590.0], "violation_rate": [0.8, 0.82, 0.78]},
    }
    latex = generate_latex_table(results)

    assert "\\begin{table}" in latex
    assert "\\end{table}" in latex
    assert "A0: UA-HCMARL" in latex
