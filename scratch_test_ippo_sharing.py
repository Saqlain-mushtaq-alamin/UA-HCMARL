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
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)
        self.reset_buffers()

    def reset_buffers(self):
        self.buffer = {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}

    def select_actions(self, obs_dict, evaluate=False, step_count=0):
        actions = {}
        with torch.no_grad():
            obs_list = [obs_dict[aid] for aid in self.agent_ids if aid in obs_dict]
            obs_arr = np.array(obs_list, dtype=np.float32)
            obs_t = torch.from_numpy(obs_arr).to(self.device)
            logits = self.actor(obs_t)
            dist = Categorical(logits=logits)
            if evaluate:
                acts = torch.argmax(logits, dim=-1).cpu().numpy()
            else:
                act_t = dist.sample()
                lp_t = dist.log_prob(act_t)
                val_t = self.critic(obs_t).squeeze(-1)
                acts = act_t.cpu().numpy()
                self.buffer["obs"].extend(obs_list)
                self.buffer["actions"].extend(acts.tolist())
                self.buffer["log_probs"].extend(lp_t.cpu().numpy().tolist())
                self.buffer["values"].extend(val_t.cpu().numpy().tolist())
            for idx, aid in enumerate(self.agent_ids):
                actions[aid] = int(acts[idx])
        return actions

    def store_transition(self, rewards, dones):
        for aid in self.agent_ids:
            if aid in rewards:
                self.buffer["rewards"].append(rewards[aid])
                self.buffer["dones"].append(dones[aid])

    def update(self, num_epochs=4):
        if len(self.buffer["obs"]) == 0:
            return {}
        obs = torch.tensor(np.array(self.buffer["obs"]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(self.buffer["actions"]), dtype=torch.long, device=self.device)
        old_lp = torch.tensor(np.array(self.buffer["log_probs"]), dtype=torch.float32, device=self.device)
        rewards = np.array(self.buffer["rewards"])
        dones = np.array(self.buffer["dones"])
        values = np.array(self.buffer["values"])

        # Compute GAE
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_val = values[t+1] if t+1 < len(rewards) else 0.0
            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + values
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(num_epochs):
            logits = self.actor(obs)
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            val = self.critic(obs).squeeze(-1)

            ratio = torch.exp(new_lp - old_lp)
            s1 = ratio * adv_t
            s2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            p_loss = -torch.min(s1, s2).mean()
            v_loss = nn.functional.mse_loss(val, ret_t)
            loss = p_loss + self.value_coef * v_loss - self.entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
            self.optimizer.step()

        self.reset_buffers()
        return {"policy_loss": p_loss.item(), "value_loss": v_loss.item()}

env = SUMOEnv(max_steps=200, seed=42)
agent_ids = [f"intersection_{i}" for i in range(16)]
ctrl = SharedIPPOController(agent_ids)
trainer = Trainer(env)

print("Training Shared IPPO for 25 episodes...")
for ep in range(25):
    summary, loss = trainer.train_episode(ctrl, max_steps=200, seed=42 + ep)
    if (ep+1) % 5 == 0:
        print(f"Ep {ep+1:2d} | Train Queue: {summary['avg_queue_length']:.2f}")

# Now test deterministic evaluation (argmax)
eval_res = trainer.run_evaluation_episode(ctrl, max_steps=200, seed=100)
print(f"\nEvaluation (argmax): Queue = {eval_res['avg_queue_length']:.2f} veh, Viol = {eval_res['constraint_violation_rate']*100:.1f}%")
