import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from src.envs.sumo_env import SUMOEnv
from src.training.trainer import Trainer

class SharedIPPOController:
    def __init__(self, agent_ids, obs_dim=16, action_dim=4, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, entropy_coef=0.01, value_coef=0.5, device="cpu"):
        self.agent_ids = agent_ids
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, action_dim)
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1)
        ).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.reset_buffers()

    def reset_buffers(self):
        self.buffer = {
            "obs": {aid: [] for aid in self.agent_ids},
            "actions": {aid: [] for aid in self.agent_ids},
            "log_probs": {aid: [] for aid in self.agent_ids},
            "rewards": {aid: [] for aid in self.agent_ids},
            "dones": {aid: [] for aid in self.agent_ids},
            "values": {aid: [] for aid in self.agent_ids},
        }

    def select_actions(self, obs_dict, evaluate=False, step_count=0):
        actions = {}
        with torch.no_grad():
            obs_list = [obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32)) for aid in self.agent_ids]
            obs_arr = np.array(obs_list, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).to(self.device)
            self.actor.eval()
            dist = Categorical(logits=self.actor(obs_t))
            self.critic.eval()
            vals = self.critic(obs_t).squeeze(-1).cpu().numpy()

            if evaluate:
                acts = torch.argmax(dist.logits, dim=-1).cpu().numpy()
            else:
                act_t = dist.sample()
                lp_t = dist.log_prob(act_t)
                acts = act_t.cpu().numpy()
                lps = lp_t.cpu().numpy()
                for idx, aid in enumerate(self.agent_ids):
                    self.buffer["obs"][aid].append(obs_list[idx])
                    self.buffer["actions"][aid].append(int(acts[idx]))
                    self.buffer["log_probs"][aid].append(float(lps[idx]))
                    self.buffer["values"][aid].append(float(vals[idx]))
            for idx, aid in enumerate(self.agent_ids):
                actions[aid] = int(acts[idx])
        return actions

    def store_transition(self, rewards, dones):
        for aid in self.agent_ids:
            if aid in rewards:
                self.buffer["rewards"][aid].append(rewards[aid])
                self.buffer["dones"][aid].append(dones[aid])

    def update(self, num_epochs=4):
        total_p_loss = 0.0
        total_v_loss = 0.0
        
        all_obs = []
        all_acts = []
        all_old_lps = []
        all_advs = []
        all_rets = []

        for aid in self.agent_ids:
            if len(self.buffer["obs"][aid]) == 0:
                continue
            obs = np.array(self.buffer["obs"][aid])
            actions = np.array(self.buffer["actions"][aid])
            old_lp = np.array(self.buffer["log_probs"][aid])
            rewards = np.array(self.buffer["rewards"][aid])
            dones = np.array(self.buffer["dones"][aid])
            values = np.array(self.buffer["values"][aid])

            num_steps = len(rewards)
            advantages = np.zeros_like(rewards, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(num_steps)):
                next_val = values[t+1] if t+1 < num_steps else 0.0
                next_non_terminal = 1.0 - float(dones[t])
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

            returns = advantages + values
            all_obs.append(obs)
            all_acts.append(actions)
            all_old_lps.append(old_lp)
            all_advs.append(advantages)
            all_rets.append(returns)

        if len(all_obs) == 0:
            return {}

        obs_t = torch.tensor(np.concatenate(all_obs, axis=0), dtype=torch.float32, device=self.device)
        acts_t = torch.tensor(np.concatenate(all_acts, axis=0), dtype=torch.long, device=self.device)
        old_lp_t = torch.tensor(np.concatenate(all_old_lps, axis=0), dtype=torch.float32, device=self.device)
        adv_t = torch.tensor(np.concatenate(all_advs, axis=0), dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(np.concatenate(all_rets, axis=0), dtype=torch.float32, device=self.device)

        if adv_t.numel() > 1 and float(adv_t.std().item()) > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        self.actor.train()
        self.critic.train()

        for _ in range(num_epochs):
            dist = Categorical(logits=self.actor(obs_t))
            new_lp = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_lp_t)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            p_loss = -torch.min(s1, s2).mean() - self.entropy_coef * entropy

            self.actor_optimizer.zero_grad()
            p_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()

            v_pred = self.critic(obs_t).squeeze(-1)
            v_loss = nn.functional.mse_loss(v_pred, ret_t)
            self.critic_optimizer.zero_grad()
            v_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()

        self.reset_buffers()
        return {"policy_loss": p_loss.item(), "value_loss": v_loss.item()}

env = SUMOEnv(max_steps=500, seed=42)
agent_ids = [f"intersection_{i}" for i in range(16)]
ctrl = SharedIPPOController(agent_ids)
trainer = Trainer(env)

print("Training Shared IPPO for 50 episodes (500 steps/ep)...")
for ep in range(50):
    summary, loss = trainer.train_episode(ctrl, max_steps=500, seed=42 + ep)
    if (ep+1) % 10 == 0:
        print(f"Ep {ep+1:2d} | Train Queue: {summary['avg_queue_length']:.2f}")

eval_res = trainer.run_evaluation_episode(ctrl, max_steps=500, seed=100)
print(f"\nEvaluation (argmax): Queue = {eval_res['avg_queue_length']:.2f} veh, Viol = {eval_res['constraint_violation_rate']*100:.1f}%")
