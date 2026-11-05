"""
Test how starvation cost registers and activates lambda during training.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

def test_training_starvation():
    num_intersections = 16
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    env = SUMOEnv(num_intersections=num_intersections, max_steps=400, max_red_wait=38, seed=44)
    
    # Set limit to 0.001 (0.1%) so even small starvation activates lambda
    ctrl = MAPPOLCEController(
        agent_ids=agent_ids,
        num_constraints=4,
        constraint_limits=[0.3, 0.3, 0.3, 0.001],
        kp=0.05,
        ki=0.005,
        kd=0.01,
    )
    
    # Load mappo_s44
    ckpt = torch.load("experiments/phase2_flat_baselines/models/mappo_s44.pt", weights_only=False)
    ctrl.shared_actor.load_state_dict(ckpt["shared_actor"])
    ctrl.cost_critic.load_state_dict(ckpt["critic"]) # initialize critic
    
    # Inject transitions that mirror the greedy evaluation where starvation happens
    trainer = Trainer(env=env)
    
    print("Pre-evaluation of unconstrained policy:")
    eval_res = trainer.run_evaluation_episode(ctrl, max_steps=400, seed=44)
    print(f"  Unconstrained MAPPO s44: Queue={eval_res['avg_queue_length']:.2f}, StarveRate={eval_res.get('max_wait_rate', 0)*100:.2f}%")
    
    # Train for 5 episodes
    for ep in range(5):
        summary, loss = trainer.train_episode(ctrl, max_steps=400, seed=44 + ep)
        print(f"  Ep {ep+1}: Queue={summary['avg_queue_length']:.2f}, MaxWaitViolations={summary.get('max_wait_violations', 0)}, Lambda_starve={loss.get('lambda_starve', 0.0):.4f}")

if __name__ == "__main__":
    test_training_starvation()
