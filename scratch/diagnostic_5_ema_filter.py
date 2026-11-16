import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.training.trainer import Trainer
from experiment_config import get_config

class EMAFilter:
    def __init__(self, ctrl, alpha=0.5):
        self.c = ctrl
        self.a = alpha
        self.s = {}

    def reset(self, ids=None):
        self.s = {}
        if hasattr(self.c, "reset"):
            if ids is not None:
                try:
                    self.c.reset(ids)
                except TypeError:
                    self.c.reset()
            else:
                self.c.reset()

    def select_actions(self, obs, evaluate=True, step_count=0, action_masks=None):
        for k, o in obs.items():
            self.s[k] = o if k not in self.s else self.a * o + (1 - self.a) * self.s[k]
        try:
            return self.c.select_actions(dict(self.s), evaluate=evaluate, step_count=step_count, action_masks=action_masks)
        except TypeError:
            return self.c.select_actions(dict(self.s), evaluate=evaluate, step_count=step_count)

def main():
    print("=" * 80)
    print("DIAGNOSTIC 5: TRIVIAL FILTER BASELINE (EMA FILTER)")
    print("=" * 80)

    cfg = get_config("canonical")
    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]

    eval_seeds = [100, 101]
    alphas = [0.2, 0.5, 0.8]
    scenarios = ["clean", "moderate", "severe", "extreme"]

    # Load clean IPPO and FMA2C
    m1_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
    m1_ippo.load_model("experiments/phase2_canonical_v2/models/ippo.pt")

    m4_fma2c = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
    m4_fma2c.load_model("experiments/phase2_canonical_v2/models/fma2c.pt")

    for sc in scenarios:
        print(f"\n================ Scenario: {sc.upper()} ================")
        
        # Test plain M1 and plain M4
        for name, ctrl in [("IPPO (No Filter)", m1_ippo), ("FMA2C (No Filter)", m4_fma2c)]:
            delays = []
            for s in eval_seeds:
                env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=s)
                trainer = Trainer(env=env)
                if hasattr(ctrl, "reset"): ctrl.reset(agent_ids)
                res = trainer.run_evaluation_episode(ctrl, max_steps=cfg.max_steps, seed=s)
                delays.append(res["avg_delay"])
            print(f"  {name:25s}: Mean = {np.mean(delays):.2f} (runs: {[round(x, 2) for x in delays]})")

        # Test EMA on IPPO
        for a in alphas:
            ema_ippo = EMAFilter(m1_ippo, alpha=a)
            delays = []
            for s in eval_seeds:
                env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=s)
                trainer = Trainer(env=env)
                ema_ippo.reset(agent_ids)
                res = trainer.run_evaluation_episode(ema_ippo, max_steps=cfg.max_steps, seed=s)
                delays.append(res["avg_delay"])
            print(f"  IPPO + EMA (alpha={a:.1f})     : Mean = {np.mean(delays):.2f} (runs: {[round(x, 2) for x in delays]})")

if __name__ == "__main__":
    main()

