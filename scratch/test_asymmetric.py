"""
Diagnostic test under asymmetric morning_peak demand to observe starvation dynamics.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.fixed_time import FixedTimeController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

def test_asymmetric():
    num_int = 4
    agent_ids = [f"intersection_{i}" for i in range(num_int)]
    
    # 1. Fixed-Time
    env_ft = SUMOEnv(num_intersections=num_int, demand_profile="morning_peak", max_steps=400, max_red_wait=38, seed=43)
    ctrl_ft = FixedTimeController(green_duration=30, yellow_duration=3)
    trainer_ft = Trainer(env=env_ft)
    ft_res = trainer_ft.run_evaluation_episode(ctrl_ft, max_steps=400, seed=43)
    print(f"Fixed-Time: Queue={ft_res['avg_queue_length']:.2f}, StarveRate={ft_res.get('max_wait_rate', 0)*100:.2f}%")

    # 2. MAPPO
    env_m = SUMOEnv(num_intersections=num_int, demand_profile="morning_peak", max_steps=400, max_red_wait=38, seed=43)
    ctrl_m = MAPPOController(agent_ids=agent_ids)
    trainer_m = Trainer(env=env_m)
    for ep in range(5):
        m_res, _ = trainer_m.train_episode(ctrl_m, max_steps=400, seed=43 + ep)
        print(f"MAPPO ep {ep+1}: Queue={m_res['avg_queue_length']:.2f}, StarveRate={m_res.get('max_wait_rate', 0)*100:.2f}%")

if __name__ == "__main__":
    test_asymmetric()
