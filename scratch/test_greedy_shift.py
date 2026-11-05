"""
Demonstrate greedy argmax shift under constrained training.
"""
import sys
sys.path.insert(0, ".")

import torch
from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

def test_greedy_shift():
    num_intersections = 4
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    env = SUMOEnv(num_intersections=num_intersections, max_steps=300, max_red_wait=38, seed=42)
    
    ctrl = MAPPOLCEController(
        agent_ids=agent_ids,
        num_constraints=4,
        constraint_limits=[0.3, 0.3, 0.3, 0.005],
        lr=2e-3,
        kp=0.1,
        ki=0.01,
        kd=0.02,
        entropy_coef=0.001,
    )
    
    with torch.no_grad():
        ctrl.shared_actor.net[-1].bias.data[0] = 1.0
        ctrl.shared_actor.net[-1].bias.data[1] = -1.0
        ctrl.shared_actor.net[-1].bias.data[2] = 0.5
        ctrl.shared_actor.net[-1].bias.data[3] = -1.0
        
    trainer = Trainer(env=env)
    
    eval_pre = trainer.run_evaluation_episode(ctrl, max_steps=300, seed=42)
    print(f"Pre-eval: Queue={eval_pre['avg_queue_length']:.2f} | StarveRate={eval_pre.get('max_wait_rate', 0)*100:.2f}% | MaxWaitEvents={eval_pre.get('max_wait_violations', 0)}")
    
    for ep in range(12):
        summary, loss = trainer.train_episode(ctrl, max_steps=300, seed=42 + ep)
        print(f"  Ep {ep+1:2d} | TrainStarve={summary.get('max_wait_rate', 0)*100:5.2f}% | Lambda_starve={loss.get('lambda_starve', 0.0):.4f}")
        
    eval_post = trainer.run_evaluation_episode(ctrl, max_steps=300, seed=42)
    print(f"Post-eval: Queue={eval_post['avg_queue_length']:.2f} | StarveRate={eval_post.get('max_wait_rate', 0)*100:.2f}% | MaxWaitEvents={eval_post.get('max_wait_violations', 0)}")
    print(f"Final Lambdas: {ctrl.pid_lagrangian.lambdas}")

if __name__ == "__main__":
    test_greedy_shift()
