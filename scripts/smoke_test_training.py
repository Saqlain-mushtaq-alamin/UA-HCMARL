"""
SMOKE TEST ONLY — NOT FOR REPORTABLE BENCHMARK METRICS
======================================================
This script runs a fast 50-episode x 500-step functional sanity check to verify
training loss backprop and convergence mechanics without running multi-hour simulations.
DO NOT use the output numbers for reporting or publication tables.
"""
import sys; sys.path.insert(0, '.')
import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.fixed_time import FixedTimeController
from src.baselines.actuated import ActuatedController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

n = 4
agent_ids = [f'intersection_{i}' for i in range(n)]
MAX_STEPS = 500  # Intentionally truncated for fast CI smoke testing only
TRAIN_EPS = 50

# Baseline reference
env_ref = SUMOEnv(num_intersections=n, max_steps=MAX_STEPS, seed=100)
trainer_ref = Trainer(env=env_ref)
ft_delay = np.mean([
    trainer_ref.run_evaluation_episode(FixedTimeController(), max_steps=MAX_STEPS, seed=100+s)['avg_delay']
    for s in range(5)
])
act_delay = np.mean([
    trainer_ref.run_evaluation_episode(ActuatedController(), max_steps=MAX_STEPS, seed=100+s)['avg_delay']
    for s in range(5)
])
print(f"REFERENCE: Fixed-Time={ft_delay:.2f}  Actuated={act_delay:.2f}")

for name, CtrlClass, kwargs in [
    ("MAPPO",     MAPPOController,    {}),
    ("MAPPO-LCE", MAPPOLCEController, {}),
]:
    env = SUMOEnv(num_intersections=n, max_steps=MAX_STEPS, seed=42)
    trainer = Trainer(env=env)
    ctrl = CtrlClass(agent_ids=agent_ids, lr=3e-4, **kwargs)
    delays = []
    for ep in range(TRAIN_EPS):
        summary, loss = trainer.train_episode(ctrl, max_steps=MAX_STEPS, seed=42+ep)
        delays.append(summary['avg_delay'])

    # Evaluate
    env_eval = SUMOEnv(num_intersections=n, max_steps=MAX_STEPS, seed=100)
    trainer_eval = Trainer(env=env_eval)
    eval_delays = [
        trainer_eval.run_evaluation_episode(ctrl, max_steps=MAX_STEPS, seed=100+s)['avg_delay']
        for s in range(5)
    ]
    eval_mean = np.mean(eval_delays)
    beats_ft = eval_mean < ft_delay
    print(f"{name}: train_last5_avg={np.mean(delays[-5:]):.2f}  eval_5ep_avg={eval_mean:.2f}  beats_fixed_time={'YES' if beats_ft else 'NO'}")

print("Smoke test done")
