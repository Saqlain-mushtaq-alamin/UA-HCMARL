import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

env = SUMOEnv(max_steps=200, seed=42)
agent_ids = [f"intersection_{i}" for i in range(16)]
ctrl = MAPPOLCEController(agent_ids)
trainer = Trainer(env)

print("Starting 5 test episodes with current MAPPO-LCE...")
for ep in range(5):
    summary, loss = trainer.train_episode(ctrl, max_steps=200, seed=42 + ep)
    print(f"Ep {ep}: viol_rate={summary['constraint_violation_rate']*100:.1f}%, "
          f"l_green={loss.get('lambda_green', 0.0):.6f}, "
          f"l_gskip={loss.get('lambda_gskip', 0.0):.6f}, "
          f"l_pskip={loss.get('lambda_pskip', 0.0):.6f}")
