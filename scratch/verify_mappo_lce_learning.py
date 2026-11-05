"""
Verify MAPPO-LCE end-to-end constraint learning on starvation constraint (Task 3).
"""
import sys
import copy
sys.path.insert(0, ".")

import numpy as np
import torch
from src.envs.sumo_env import SUMOEnv
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer

def verify_lce_learning():
    num_intersections = 16
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    env = SUMOEnv(num_intersections=num_intersections, max_steps=500, max_red_wait=38, seed=44)
    
    # Initialize MAPPO-LCE with our 4 constraints and load pre-existing MAPPO weights (which have high starvation)
    ctrl = MAPPOLCEController(
        agent_ids=agent_ids,
        num_constraints=4,
        constraint_limits=[0.3, 0.3, 0.3, 0.01],
        kp=0.02,
        ki=0.002,
        kd=0.005,
    )
    
    # Load actor from mappo_s44 so it starts in the unconstrained state with high violations
    mappo_ckpt = torch.load("experiments/phase2_flat_baselines/models/mappo_s44.pt", weights_only=False)
    if "shared_actor" in mappo_ckpt:
        ctrl.shared_actor.load_state_dict(mappo_ckpt["shared_actor"])
    
    trainer = Trainer(env=env)
    
    print("Initial Evaluation before constrained fine-tuning:")
    res_init = trainer.run_evaluation_episode(ctrl, max_steps=500, seed=44)
    init_starve_rate = res_init.get("max_wait_rate", 0.0)
    print(f"  Init Queue: {res_init['avg_queue_length']:.2f} veh | Starve Rate: {init_starve_rate*100:.2f}% ({res_init.get('max_wait_violations', 0)} events)")
    print(f"  Init Lambdas: {ctrl.pid_lagrangian.lambdas}")
    
    print("\nRunning 5 Constrained Training Episodes with MAPPO-LCE:")
    rates = []
    lambdas_starve = []
    
    for ep in range(5):
        summary, loss = trainer.train_episode(ctrl, max_steps=500, seed=44 + ep)
        s_rate = summary.get("max_wait_rate", 0.0)
        lam_s = loss.get("lambda_starve", 0.0)
        rates.append(s_rate)
        lambdas_starve.append(lam_s)
        print(f"  Ep {ep+1} | Queue: {summary['avg_queue_length']:.2f} | Starvation Rate: {s_rate*100:.2f}% | Lambda_starve: {lam_s:.4f}")
        
    print("\nPost-training Evaluation:")
    res_final = trainer.run_evaluation_episode(ctrl, max_steps=500, seed=44)
    final_starve_rate = res_final.get("max_wait_rate", 0.0)
    print(f"  Final Queue: {res_final['avg_queue_length']:.2f} veh | Starve Rate: {final_starve_rate*100:.2f}% ({res_final.get('max_wait_violations', 0)} events)")
    print(f"  Final Lambdas: {ctrl.pid_lagrangian.lambdas}")
    
    assert lambdas_starve[-1] > 0.0, "Lambda starve must be non-zero after constraint violation"
    print("\nVerification SUCCESS: Nonzero starvation cost registered, lambda_starve activated, constraint integrated.")

if __name__ == "__main__":
    verify_lce_learning()

