"""
Demonstrate MAPPO-LCE adapting against starvation violations when policy attempts to hold green.
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import torch
from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

def test_constrained_adaptation():
    num_intersections = 4
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    env = SUMOEnv(num_intersections=num_intersections, max_steps=400, max_red_wait=38, seed=42)
    
    ctrl = MAPPOLCEController(
        agent_ids=agent_ids,
        num_constraints=4,
        constraint_limits=[0.3, 0.3, 0.3, 0.005],
        kp=0.05,
        ki=0.005,
        kd=0.01,
        entropy_coef=0.001,  # low entropy so learned preference dominates
    )
    
    # Bias the output layer of the shared actor strongly toward Phase 0 (hold NS green)
    # This simulates a policy that learned to hold phase 0 for traffic queue rewards
    with torch.no_grad():
        ctrl.shared_actor.net[-1].bias.data[0] = 3.0
        ctrl.shared_actor.net[-1].bias.data[1] = -1.0
        ctrl.shared_actor.net[-1].bias.data[2] = 0.0
        ctrl.shared_actor.net[-1].bias.data[3] = -1.0
        
    trainer = Trainer(env=env)
    
    print("Pre-training greedy evaluation (biased policy holding Phase 0):")
    eval_pre = trainer.run_evaluation_episode(ctrl, max_steps=400, seed=42)
    print(f"  Pre-eval: Queue={eval_pre['avg_queue_length']:.2f} | Starvation Rate={eval_pre.get('max_wait_rate', 0)*100:.2f}% ({eval_pre.get('max_wait_violations', 0)} events)")
    print(f"  Initial lambdas: {ctrl.pid_lagrangian.lambdas}")
    
    print("\nTraining MAPPO-LCE with Starvation Penalty:")
    for ep in range(6):
        summary, loss = trainer.train_episode(ctrl, max_steps=400, seed=42 + ep)
        s_rate = summary.get("max_wait_rate", 0.0)
        lam_s = loss.get("lambda_starve", 0.0)
        print(f"  Ep {ep+1} | Queue: {summary['avg_queue_length']:.2f} | Train StarveRate: {s_rate*100:.2f}% | Lambda_starve: {lam_s:.4f}")
        
    print("\nPost-training greedy evaluation:")
    eval_post = trainer.run_evaluation_episode(ctrl, max_steps=400, seed=42)
    print(f"  Post-eval: Queue={eval_post['avg_queue_length']:.2f} | Starvation Rate={eval_post.get('max_wait_rate', 0)*100:.2f}% ({eval_post.get('max_wait_violations', 0)} events)")
    print(f"  Final lambdas: {ctrl.pid_lagrangian.lambdas}")

if __name__ == "__main__":
    test_constrained_adaptation()
