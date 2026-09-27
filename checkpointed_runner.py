"""
checkpointed_runner.py — Resumable, Crash-Proof Experiment Evaluation Engine
=============================================================================
Provides atomic on-disk checkpointing for multi-seed, multi-scenario evaluation loops.
If power fails or the process is interrupted, re-running the exact same script will:
  1. Inspect the existing JSON output file.
  2. Skip all (scenario, method, seed) runs that are already recorded.
  3. Resume execution from the first missing run.
  4. Write every new evaluation atomically via a temporary file + replace to prevent corruption.

EXAMPLE USAGE:
--------------
from checkpointed_runner import ResultCheckpointer

checkpointer = ResultCheckpointer("experiments/phase5_ablations/phase5_raw_results.json")

for noise in noise_scenarios:
    for seed in range(num_seeds):
        for method_name, ctrl in ablations.items():
            if checkpointer.is_completed(noise, method_name, seed):
                # Skip already computed results
                continue
            
            # Run evaluation
            summary = trainer_eval.run_evaluation_episode(ctrl, max_steps=max_steps, seed=ep_seed)
            
            # Checkpoint atomically to disk
            checkpointer.record_result(
                scenario=noise,
                method=method_name,
                avg_delay=summary["avg_delay"],
                violation_rate=summary["constraint_violation_rate"],
            )

results = checkpointer.get_results()
"""

import os
import json
import tempfile
from typing import Dict, Any, Optional, List


class ResultCheckpointer:
    def __init__(self, filepath: str, scenarios: Optional[List[str]] = None, methods: Optional[List[str]] = None):
        self.filepath = os.path.abspath(filepath)
        self.scenarios = scenarios or ["clean", "mild", "moderate", "severe", "extreme"]
        self.methods = methods or []
        self.results: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.load()

    def load(self):
        """Loads existing results if file exists and initializes structure."""
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self.results = data
                    # Ensure nested dictionary structure
                    for s in self.scenarios:
                        if s not in self.results:
                            self.results[s] = {}
                        for m in self.methods:
                            if m not in self.results[s]:
                                self.results[s][m] = {"avg_delay": [], "violation_rate": []}
                    return
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Checkpoint file {self.filepath} corrupted or unreadable ({e}). Initializing clean state.")

        # Initialize fresh structure
        self.results = {
            s: {m: {"avg_delay": [], "violation_rate": []} for m in self.methods}
            for s in self.scenarios
        }

    def is_completed(self, scenario: str, method: str, seed_index: int) -> bool:
        """Returns True if the specific seed index for (scenario, method) has already been saved."""
        if scenario not in self.results or method not in self.results[scenario]:
            return False
        rec = self.results[scenario][method]
        delays = rec.get("avg_delay", [])
        viols = rec.get("violation_rate", [])
        # Both must be present and length must exceed the requested seed index
        return len(delays) > seed_index and len(viols) > seed_index

    def record_result(self, scenario: str, method: str, avg_delay: float, violation_rate: float):
        """Appends result and immediately performs an atomic flush to disk."""
        if scenario not in self.results:
            self.results[scenario] = {}
        if method not in self.results[scenario]:
            self.results[scenario][method] = {"avg_delay": [], "violation_rate": []}

        self.results[scenario][method]["avg_delay"].append(float(avg_delay))
        self.results[scenario][method]["violation_rate"].append(float(violation_rate))

        self.atomic_save()

    def atomic_save(self):
        """Atomic write using temporary file + atomic rename to prevent corruption during sudden power cut."""
        dir_name = os.path.dirname(self.filepath)
        os.makedirs(dir_name, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(dir=dir_name, prefix="ckpt_", suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.filepath)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_results(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        return self.results

    def get_completed_count(self, scenario: str, method: str) -> int:
        if scenario in self.results and method in self.results[scenario]:
            return len(self.results[scenario][method].get("avg_delay", []))
        return 0
