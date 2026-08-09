"""
Staged Curriculum Scheduler (Phase 1 §1.3 & Blueprint §7)

Phases:
Phase A: Pre-train BSE supervised (~50K steps)
Phase B: Train workers (BSE frozen) (~200K steps)
Phase C: Joint fine-tune BSE + workers (~200K steps)
Phase D: Add manager, full system (~500K steps)
"""

from typing import Dict


class CurriculumScheduler:
    """
    Manages multi-stage training curriculum across training phases A, B, C, D.
    """

    def __init__(
        self,
        phase_a_steps: int = 50000,
        phase_b_steps: int = 200000,
        phase_c_steps: int = 200000,
        phase_d_steps: int = 500000,
    ):
        self.phase_a_steps = phase_a_steps
        self.phase_b_steps = phase_b_steps
        self.phase_c_steps = phase_c_steps
        self.phase_d_steps = phase_d_steps

        self.threshold_a = phase_a_steps
        self.threshold_b = self.threshold_a + phase_b_steps
        self.threshold_c = self.threshold_b + phase_c_steps
        self.threshold_d = self.threshold_c + phase_d_steps

    def get_phase(self, step: int) -> Dict[str, bool]:
        """
        Determines component freeze/train flags for total step count.
        """
        if step < self.threshold_a:
            return {"phase": "A", "train_bse": True, "train_worker": False, "train_manager": False}
        elif step < self.threshold_b:
            return {"phase": "B", "train_bse": False, "train_worker": True, "train_manager": False}
        elif step < self.threshold_c:
            return {"phase": "C", "train_bse": True, "train_worker": True, "train_manager": False}
        else:
            return {"phase": "D", "train_bse": True, "train_worker": True, "train_manager": True}
