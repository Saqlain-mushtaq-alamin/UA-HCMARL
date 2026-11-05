"""
Diagnostic script to test constraint learning across Fixed-Time, MAPPO, and MAPPO-LCE (Task 3).
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, ".")

from src.envs.sumo_env import SUMOEnv
from src.baselines.fixed_time import FixedTimeController
from src.baselines.mappo import MAPPOController
from src.baselines.mappo_lce import MAPPOLCEController
from src.training.trainer import Trainer, LearningCurveLogger


def run_diagnostic():
    num_intersections = 4
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    max_steps = 400
    train_episodes = 8
    seed = 42

    print("==================================================")
    print("Task 3 Diagnostic: Constraint Learning Validation")
    print(f"Intersections: {num_intersections} | Steps/ep: {max_steps} | Eps: {train_episodes}")
    print("==================================================")

    # 1. Fixed-Time Evaluation
    env_ft = SUMOEnv(num_intersections=num_intersections, max_steps=max_steps, max_red_wait=38, seed=seed)
    ctrl_ft = FixedTimeController(cycle_length=60, min_green=5, yellow_time=3)
    trainer_ft = Trainer(env=env_ft)
    ft_summary = trainer_ft.run_evaluation_episode(ctrl_ft, max_steps=max_steps, seed=seed)
    ft_starve_rate = ft_summary.get("max_wait_rate", 0.0)
    print(f"\n[Fixed-Time Baseline]")
    print(f"  Queue: {ft_summary['avg_queue_length']:.2f} veh")
    print(f"  Starvation (max_wait) Rate: {ft_starve_rate * 100:.2f}%")
    print(f"  Violations: {ft_summary.get('max_wait_violations', 0)} events")

    # 2. Unconstrained MAPPO Training
    print(f"\n[Unconstrained MAPPO Training ({train_episodes} eps)]")
    env_mappo = SUMOEnv(num_intersections=num_intersections, max_steps=max_steps, max_red_wait=38, seed=seed)
    ctrl_mappo = MAPPOController(agent_ids=agent_ids)
    logger_mappo = LearningCurveLogger()
    trainer_mappo = Trainer(env=env_mappo, learning_curve_logger=logger_mappo)
    
    mappo_starve_rates = []
    for ep in range(train_episodes):
        summary, loss = trainer_mappo.train_episode(ctrl_mappo, max_steps=max_steps, seed=seed + ep)
        s_rate = summary.get("max_wait_rate", 0.0)
        mappo_starve_rates.append(s_rate)
        print(f"  Ep {ep+1:2d} | Queue: {summary['avg_queue_length']:.2f} | Starvation Rate: {s_rate * 100:.2f}%")

    # 3. Repaired MAPPO-LCE Training
    print(f"\n[Repaired MAPPO-LCE Training ({train_episodes} eps)]")
    env_lce = SUMOEnv(num_intersections=num_intersections, max_steps=max_steps, max_red_wait=38, seed=seed)
    # Target starvation limit = 0.005 (0.5%), so positive starvation will activate lambda
    ctrl_lce = MAPPOLCEController(
        agent_ids=agent_ids,
        num_constraints=4,
        constraint_limits=[0.3, 0.3, 0.3, 0.005],
        kp=0.02,
        ki=0.002,
        kd=0.005,
    )
    logger_lce = LearningCurveLogger()
    trainer_lce = Trainer(env=env_lce, learning_curve_logger=logger_lce)

    lce_starve_rates = []
    lce_lambdas = []
    for ep in range(train_episodes):
        summary, loss = trainer_lce.train_episode(ctrl_lce, max_steps=max_steps, seed=seed + ep)
        s_rate = summary.get("max_wait_rate", 0.0)
        lam_starve = loss.get("lambda_starve", 0.0)
        lce_starve_rates.append(s_rate)
        lce_lambdas.append(lam_starve)
        print(f"  Ep {ep+1:2d} | Queue: {summary['avg_queue_length']:.2f} | Starvation Rate: {s_rate * 100:.2f}% | Lambda_starve: {lam_starve:.4f}")

    print("\n================== SUMMARY ==================")
    print(f"Fixed-Time Starvation Rate: {ft_starve_rate * 100:.2f}% (Expected < 0.5%)")
    print(f"MAPPO Mean Starvation Rate: {np.mean(mappo_starve_rates) * 100:.2f}%")
    print(f"MAPPO-LCE Final Starvation Rate: {lce_starve_rates[-1] * 100:.2f}%")
    print(f"MAPPO-LCE Final Lambda_starve: {lce_lambdas[-1]:.4f}")
    print("=============================================")

if __name__ == "__main__":
    run_diagnostic()
