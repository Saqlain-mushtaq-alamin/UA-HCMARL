import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.baselines.fma2c import FMA2CController
from src.training.trainer import Trainer

class Blind:
    def __init__(self, ctrl):
        self.c = ctrl
    def reset(self, ids=None):
        if hasattr(self.c, "reset"):
            self.c.reset(ids)
    def select_actions(self, obs, evaluate=True, step_count=0):
        z = {a: np.zeros_like(o) for a, o in obs.items()}
        return self.c.select_actions(z, evaluate=evaluate, step_count=step_count)

env = SUMOEnv(max_steps=500, seed=42)
trainer = Trainer(env)
agent_ids = [f"intersection_{i}" for i in range(16)]

configs = [
    ("MAPPO", MAPPOController, "experiments/phase2_flat_baselines/models/mappo.pt"),
    ("MAPPO-LCE", MAPPOLCEController, "experiments/phase2_flat_baselines/models/mappo_lce.pt"),
    ("FMA2C", lambda aids: FMA2CController(aids, num_regions=4), "experiments/phase2_flat_baselines/models/fma2c.pt"),
]

for name, ctrl_factory, path in configs:
    ctrl = ctrl_factory(agent_ids)
    ctrl.load_model(path)
    res_norm = trainer.run_evaluation_episode(ctrl, max_steps=500, seed=42)
    blind_ctrl = Blind(ctrl)
    res_blind = trainer.run_evaluation_episode(blind_ctrl, max_steps=500, seed=42)
    q_norm = res_norm["avg_queue_length"]
    v_norm = res_norm["constraint_violation_rate"] * 100
    q_blind = res_blind["avg_queue_length"]
    v_blind = res_blind["constraint_violation_rate"] * 100
    print(f"{name:10s} | Normal: Queue={q_norm:6.2f}, Viol={v_norm:5.1f}% | Blind: Queue={q_blind:6.2f}, Viol={v_blind:5.1f}%")
