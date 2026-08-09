"""
Experiment Logging & Checkpointing Callbacks (Phase 1 §1.3)

Provides structured file logging and optional Weights & Biases (wandb) logging.
"""

import json
import os
from typing import Dict, Any, Optional

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """
    Handles step/episode logging to JSON files and WandB dashboard.
    """

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "experiments",
        use_wandb: bool = False,
        wandb_project: str = "UA-HCMARL",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.experiment_name = experiment_name
        self.output_dir = os.path.join(output_dir, experiment_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.json_log_path = os.path.join(self.output_dir, "metrics.jsonl")
        self.use_wandb = use_wandb and WANDB_AVAILABLE

        if self.use_wandb:
            try:
                wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config,
                    reinit=True,
                )
            except Exception as e:
                print(f"[Warning] Failed to initialize wandb: {e}. Falling back to file logger.")
                self.use_wandb = False

    def log_metrics(self, step: int, metrics: Dict[str, float]):
        """Logs metrics dictionary for given step/episode."""
        log_entry = {"step": step, **metrics}
        
        # Write to JSONL
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # WandB logging
        if self.use_wandb:
            try:
                wandb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self):
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass
