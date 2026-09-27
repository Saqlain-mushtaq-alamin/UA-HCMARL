import os, sys
sys.path.insert(0, os.path.abspath("."))

from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

net_file = "networks/grid4x4/grid4x4.net.xml"
route_file = "networks/grid4x4/demand_uniform.rou.xml"
num_intersections = 16
agent_ids = [f"intersection_{i}" for i in range(num_intersections)]

env_train = SUMOEnv(net_file=net_file, route_file=route_file, num_intersections=num_intersections, max_steps=500, seed=42)
trainer = Trainer(env=env_train)

a5 = MAPPOLCEController(agent_ids=agent_ids)
a6 = MAPPOController(agent_ids=agent_ids)

print("Training A5 (MAPPO-LCE) and A6 (MAPPO) for 5 episodes at max_steps=500...")
for ep in range(5):
    trainer.train_episode(a5, max_steps=500, seed=42 + ep)
    trainer.train_episode(a6, max_steps=500, seed=42 + ep)

env_eval = SUMOEnv(net_file=net_file, route_file=route_file, num_intersections=num_intersections, max_steps=500, seed=42)
trainer_eval = Trainer(env=env_eval)
res5 = trainer_eval.run_evaluation_episode(a5, max_steps=500, seed=42)
res6 = trainer_eval.run_evaluation_episode(a6, max_steps=500, seed=42)
print(f"A5 (500 steps): avg_delay={res5['avg_delay']:.2f}s, viol={res5['constraint_violation_rate']*100:.1f}%")
print(f"A6 (500 steps): avg_delay={res6['avg_delay']:.2f}s, viol={res6['constraint_violation_rate']*100:.1f}%")
