import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.fixed_time import FixedTimeController

def run(policy, steps=500, n=16, seed=0):
    env = SUMOEnv(num_intersections=n, max_steps=steps, seed=seed)
    obs, _ = env.reset(seed=seed)
    q, tp, viol = [], 0.0, 0
    for t in range(steps):
        obs, _, _, _, info = env.step(policy(obs, t))
        q.append(np.mean(list(info["step_delays"].values())))
        tp += info["step_throughput"]
        viol += sum(info["step_violations"].values())
    return np.mean(q), tp, viol / (steps * n)

if __name__ == "__main__":
    print("=" * 60)
    print("        PREFLIGHT SIMULATOR LOOPHOLE CHECK")
    print("=" * 60)
    ft = FixedTimeController()
    policies = {f"always_{k}": (lambda o, t, k=k: {a: k for a in o}) for k in range(4)}
    policies["random"] = lambda o, t: {a: np.random.randint(4) for a in o}
    policies["fixed_time"] = lambda o, t: ft.select_actions(o, step_count=t)

    for name, pol in policies.items():
        q, tp, v = run(pol)
        print(f"{name:12s} queue={q:7.2f}  throughput={tp:8.1f}  viol/int-step={v:.3f}")
