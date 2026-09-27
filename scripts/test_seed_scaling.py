import os, sys
sys.path.insert(0, os.path.abspath("."))

from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo import MAPPOController
from src.training.trainer import Trainer

net_file = "networks/grid4x4/grid4x4.net.xml"
route_file = "networks/grid4x4/demand_uniform.rou.xml"
num_intersections = 16
agent_ids = [f"intersection_{i}" for i in range(num_intersections)]

ctrl = MAPPOController(agent_ids=agent_ids)

for seed in [0, 1, 2, 42]:
    env = SUMOEnv(net_file=net_file, route_file=route_file, num_intersections=num_intersections, max_steps=500, seed=seed)
    trainer = Trainer(env=env)
    res = trainer.run_evaluation_episode(ctrl, max_steps=500, seed=seed)
    print(f"seed={seed:2d}: delay={res['avg_delay']:6.2f}s, viol={res['constraint_violation_rate']*100:.1f}%")
