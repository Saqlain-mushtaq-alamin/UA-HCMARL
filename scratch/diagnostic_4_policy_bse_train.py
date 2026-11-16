import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath("."))
from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.models.belief_encoder import GRUKalmanNet
from src.training.trainer import Trainer
from experiment_config import get_config

# ---------------------------------------------------------------------------
# 1. Dataset Collection on Policy Rollouts
# ---------------------------------------------------------------------------
def collect_bse_dataset_from_policy(
    policy,
    env: SUMOEnv,
    num_steps: int = 2000,
    history_len: int = 5,
    noise_scenarios = None,
    seed: int = 42,
):
    if noise_scenarios is None:
        noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]

    history_samples = []
    ground_truth_samples = []
    agent_ids = env.intersection_ids
    steps_per_sc = max(1, num_steps // len(noise_scenarios))

    for idx, noise in enumerate(noise_scenarios):
        env.noise_injector.set_scenario(noise)
        obs_dict, reset_info = env.reset(seed=seed + idx * 1000)
        action_masks = reset_info.get("action_masks", None)
        agent_histories = {aid: [obs_dict[aid]] * history_len for aid in agent_ids}

        for step in range(steps_per_sc):
            actions = policy.select_actions(
                obs_dict, evaluate=False, step_count=step, action_masks=action_masks
            )
            obs_dict, _, dones, _, info = env.step(actions)
            action_masks = info.get("action_masks", None)

            for aid in agent_ids:
                noisy_obs = obs_dict[aid]
                true_obs = env._get_clean_observation(aid)
                agent_histories[aid].append(noisy_obs)
                if len(agent_histories[aid]) > history_len:
                    agent_histories[aid].pop(0)

                history_samples.append(np.array(agent_histories[aid], dtype=np.float32))
                ground_truth_samples.append(true_obs.astype(np.float32))

            if all(dones.values()):
                obs_dict, reset_info = env.reset()
                action_masks = reset_info.get("action_masks", None)
                agent_histories = {aid: [obs_dict[aid]] * history_len for aid in agent_ids}

    return np.array(history_samples, dtype=np.float32), np.array(ground_truth_samples, dtype=np.float32)

# ---------------------------------------------------------------------------
# 2. Supervised Training of BSE (Non-Recursive)
# ---------------------------------------------------------------------------
def train_bse_model(bse, X_train, Y_train, epochs=20, batch_size=64, lr=1e-3):
    bse.train()
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(bse.parameters(), lr=lr)

    for ep in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            b_hat, sigma_hat, _ = bse(bx, prev_belief=None)
            loss_dict = bse.compute_loss(b_hat, sigma_hat, by, beta=0.1)
            loss = loss_dict["loss_bse"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

# ---------------------------------------------------------------------------
# 3. Environment Wrapper that transforms observations via frozen BSE
# ---------------------------------------------------------------------------
class FrozenBSEEnvWrapper:
    """Wraps environment so agent interacts strictly via frozen-BSE outputs."""
    def __init__(self, env: SUMOEnv, bse_model: GRUKalmanNet, history_len: int = 5):
        self.env = env
        self.bse = bse_model
        self.bse.eval()
        self.history_len = history_len
        self.agent_ids = env.intersection_ids
        self.num_intersections = env.num_intersections
        self.obs_dim = 16
        self.histories = {aid: [] for aid in self.agent_ids}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed=None):
        obs_dict, reset_info = self.env.reset(seed=seed)
        self.histories = {aid: [obs_dict[aid]] * self.history_len for aid in self.agent_ids}
        denoised_obs = self._get_bse_obs()
        return denoised_obs, reset_info

    def _get_bse_obs(self):
        batch = [self.histories[aid] for aid in self.agent_ids]
        hist_t = torch.tensor(np.array(batch), dtype=torch.float32)
        with torch.no_grad():
            b_hat_t, _, _ = self.bse(hist_t, prev_belief=None)
            b_hat_np = b_hat_t.numpy()
        return {aid: b_hat_np[i] for i, aid in enumerate(self.agent_ids)}

    def step(self, actions):
        obs_dict, rewards, dones, truncateds, info = self.env.step(actions)
        for aid in self.agent_ids:
            self.histories[aid].append(obs_dict[aid])
            if len(self.histories[aid]) > self.history_len:
                self.histories[aid].pop(0)
        denoised_obs = self._get_bse_obs()
        return denoised_obs, rewards, dones, truncateds, info

    @property
    def action_space(self):
        return self.env.action_space

# ---------------------------------------------------------------------------
# 4. Main Training and Evaluation
# ---------------------------------------------------------------------------
def main():
    print("=" * 80, flush=True)
    print("DIAGNOSTIC 4: TRAIN BSE ON POLICY ROLLOUTS + TRAIN POLICY ON FROZEN-BSE", flush=True)
    print("=" * 80, flush=True)

    cfg = get_config("canonical")
    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]

    # Step 4.1: Load Clean IPPO policy to generate realistic rollouts
    print("\n[Step 4.1] Loading Clean IPPO Policy to collect rollouts...", flush=True)
    clean_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
    clean_ippo.load_model("experiments/phase2_canonical_v2/models/ippo.pt")

    # Step 4.2: Collect BSE training dataset on policy rollouts
    print("[Step 4.2] Collecting BSE dataset on policy rollouts (train seed 42, test seed 999)...", flush=True)
    env_bse = SUMOEnv(num_intersections=cfg.num_intersections, max_steps=cfg.max_steps, seed=42)
    X_train, Y_train = collect_bse_dataset_from_policy(clean_ippo, env_bse, num_steps=2000, seed=42)
    X_test, Y_test = collect_bse_dataset_from_policy(clean_ippo, env_bse, num_steps=500, seed=999)
    print(f"  Collected {len(X_train)} train samples, {len(X_test)} validation samples.", flush=True)

    # Step 4.3: Train BSE
    print("[Step 4.3] Training BSE on policy rollouts (20 epochs)...", flush=True)
    policy_bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
    train_bse_model(policy_bse, X_train, Y_train, epochs=20, batch_size=64)
    
    # Evaluate offline calibration
    policy_bse.eval()
    with torch.no_grad():
        b_hat_test, _, _ = policy_bse(torch.tensor(X_test), prev_belief=None)
        test_raw_mse = float(torch.mean((torch.tensor(X_test[:, -1, :]) - torch.tensor(Y_test)) ** 2).item())
        test_bse_mse = float(torch.mean((b_hat_test - torch.tensor(Y_test)) ** 2).item())
        mse_red = (test_raw_mse - test_bse_mse) / test_raw_mse * 100
        print(f"  Test Raw MSE: {test_raw_mse:.4f} -> BSE MSE: {test_bse_mse:.4f} ({mse_red:.1f}% reduction)", flush=True)

    # Step 4.4: Train IPPO policy on frozen-BSE outputs (Phase B)
    print("\n[Step 4.4] Training IPPO on frozen-BSE outputs across noise mixture (50 episodes)...", flush=True)
    phase_b_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
    
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    base_env = SUMOEnv(num_intersections=cfg.num_intersections, max_steps=cfg.max_steps, seed=42)
    wrapped_env = FrozenBSEEnvWrapper(base_env, policy_bse, history_len=5)
    trainer_b = Trainer(env=wrapped_env)

    t0 = time.time()
    for ep in range(50):
        noise = noise_scenarios[ep % len(noise_scenarios)]
        base_env.noise_injector.set_scenario(noise)
        trainer_b.train_episode(phase_b_ippo, max_steps=cfg.max_steps, seed=42 + ep)
        if (ep + 1) % 10 == 0:
            print(f"  Trained {ep + 1}/50 episodes ({time.time() - t0:.1f}s)...", flush=True)

    # Step 4.5: Comprehensive Evaluation across scenarios
    print("\n[Step 4.5] Comprehensive Comparative Evaluation...", flush=True)
    
    # Import EMA filter
    class EMAFilter:
        def __init__(self, ctrl, alpha=0.5):
            self.c = ctrl
            self.a = alpha
            self.s = {}
        def reset(self, ids=None):
            self.s = {}
            if hasattr(self.c, "reset"):
                try: self.c.reset(ids)
                except TypeError: self.c.reset()
        def select_actions(self, obs, evaluate=True, step_count=0, action_masks=None):
            for k, o in obs.items():
                self.s[k] = o if k not in self.s else self.a * o + (1 - self.a) * self.s[k]
            try:
                return self.c.select_actions(dict(self.s), evaluate=evaluate, step_count=step_count, action_masks=action_masks)
            except TypeError:
                return self.c.select_actions(dict(self.s), evaluate=evaluate, step_count=step_count)

    from scratch.diagnostic_3_non_recursive import NonRecursiveBSEWrapper
    
    # Load old random BSE
    old_bse = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
    old_data = torch.load("experiments/phase3_canonical/models/bse.pt", map_location="cpu", weights_only=False)
    old_bse.load_state_dict(old_data["bse_state"] if "bse_state" in old_data else old_data)
    old_bse.eval()

    eval_seeds = [100, 101]
    test_scenarios = ["clean", "moderate", "severe", "extreme"]

    for sc in test_scenarios:
        print(f"\n================ Evaluation Scenario: {sc.upper()} ================", flush=True)
        
        # 1. Plain IPPO
        plain_res = []
        for s in eval_seeds:
            env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=s)
            t = Trainer(env=env)
            clean_ippo.reset(agent_ids)
            plain_res.append(t.run_evaluation_episode(clean_ippo, max_steps=cfg.max_steps, seed=s)["avg_delay"])

        # 2. EMA Filter (alpha=0.5)
        ema_res = []
        for s in eval_seeds:
            env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=s)
            t = Trainer(env=env)
            ema = EMAFilter(clean_ippo, alpha=0.5)
            ema_res.append(t.run_evaluation_episode(ema, max_steps=cfg.max_steps, seed=s)["avg_delay"])

        # 3. Old BSE (Non-Recursive) + Clean IPPO
        old_bse_res = []
        for s in eval_seeds:
            env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=s)
            t = Trainer(env=env)
            w = NonRecursiveBSEWrapper(base_controller=clean_ippo, bse_model=old_bse, history_len=5)
            w.reset(agent_ids)
            old_bse_res.append(t.run_evaluation_episode(w, max_steps=cfg.max_steps, seed=s)["avg_delay"])

        # 4. New Policy-BSE (Non-Recursive) + Phase B Trained IPPO
        new_bse_res = []
        for s in eval_seeds:
            env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=s)
            t = Trainer(env=env)
            w = NonRecursiveBSEWrapper(base_controller=phase_b_ippo, bse_model=policy_bse, history_len=5)
            w.reset(agent_ids)
            new_bse_res.append(t.run_evaluation_episode(w, max_steps=cfg.max_steps, seed=s)["avg_delay"])

        print(f"  1. IPPO (Plain Clean)                  : {np.mean(plain_res):.2f} (runs: {[round(x, 2) for x in plain_res]})", flush=True)
        print(f"  2. IPPO + EMA Filter (alpha=0.5)       : {np.mean(ema_res):.2f} (runs: {[round(x, 2) for x in ema_res]})", flush=True)
        print(f"  3. IPPO + Random BSE (Old Non-Rec)     : {np.mean(old_bse_res):.2f} (runs: {[round(x, 2) for x in old_bse_res]})", flush=True)
        print(f"  4. IPPO (Phase B) + Policy BSE (New)   : {np.mean(new_bse_res):.2f} (runs: {[round(x, 2) for x in new_bse_res]})", flush=True)

if __name__ == "__main__":
    main()

