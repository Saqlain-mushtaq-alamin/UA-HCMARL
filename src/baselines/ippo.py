"""
Independent PPO Baseline (B3) (Phase 2 §2.2)

Independent PPO per intersection without parameter sharing.
Features:
- 2-layer MLP (128, 64) for actor and critic.
- Discrete action selection with categorical distribution.
- Standard PPO policy clip and value loss updates.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class IPPOAgent(nn.Module):
    """
    Single agent PPO actor-critic model.
    """

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        hidden_dims: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, entropy, value


class IPPOController:
    """
    Multi-Agent Independent PPO controller wrapper.
    Manages separate IPPOAgent instances for each intersection.
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.agent_ids = agent_ids
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        self.agents: Dict[str, IPPOAgent] = {}
        self.optimizers: Dict[str, optim.Optimizer] = {}

        for aid in agent_ids:
            agent = IPPOAgent(obs_dim, action_dim).to(self.device)
            self.agents[aid] = agent
            self.optimizers[aid] = optim.Adam(agent.parameters(), lr=lr)

        self.reset_buffers()

    def reset_buffers(self):
        self.buffers: Dict[str, Dict[str, List]] = {
            aid: {
                "obs": [],
                "actions": [],
                "log_probs": [],
                "rewards": [],
                "dones": [],
                "values": [],
            }
            for aid in self.agent_ids
        }

    def select_action(
        self, agent_id: str, obs: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, float, float]:
        agent = self.agents[agent_id]
        agent.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            if evaluate:
                logits = agent.actor(obs_t)
                act_t = torch.argmax(logits, dim=-1)
                log_prob_t = torch.zeros(1, device=self.device)
                val_t = agent.critic(obs_t).squeeze(-1)
            else:
                act_t, log_prob_t, _, val_t = agent.get_action_and_value(obs_t)

        return (
            int(act_t.item()),
            float(log_prob_t.item()),
            float(val_t.item()),
        )

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], evaluate: bool = False, step_count: int = 0
    ) -> Dict[str, int]:
        actions = {}
        with torch.no_grad():
            for aid in self.agent_ids:
                if aid in self.agents and aid in obs_dict:
                    agent = self.agents[aid]
                    agent.eval()
                    obs = obs_dict[aid]
                    obs_t = torch.from_numpy(obs).unsqueeze(0).to(self.device)
                    if evaluate:
                        logits = agent.actor(obs_t)
                        act = int(torch.argmax(logits, dim=-1).item())
                        actions[aid] = act
                    else:
                        act_t, lp_t, _, val_t = agent.get_action_and_value(obs_t)
                        act = int(act_t.item())
                        actions[aid] = act
                        self.buffers[aid]["obs"].append(obs)
                        self.buffers[aid]["actions"].append(act)
                        self.buffers[aid]["log_probs"].append(float(lp_t.item()))
                        self.buffers[aid]["values"].append(float(val_t.item()))
                else:
                    actions[aid] = 0
        return actions

    def save_model(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({aid: agent.state_dict() for aid, agent in self.agents.items()}, path)

    def load_model(self, path: str):
        state_dicts = torch.load(path, map_location=self.device, weights_only=False)
        for aid, agent in self.agents.items():
            if aid in state_dicts:
                agent.load_state_dict(state_dicts[aid])

    def store_transition(self, rewards: Dict[str, float], dones: Dict[str, bool]):
        for aid in self.agent_ids:
            if aid in rewards:
                self.buffers[aid]["rewards"].append(rewards[aid])
                self.buffers[aid]["dones"].append(dones[aid])

    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        losses = {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}
        num_updates = 0

        for aid in self.agent_ids:
            buf = self.buffers[aid]
            if len(buf["obs"]) == 0:
                continue

            obs = torch.tensor(np.array(buf["obs"]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(buf["actions"]), dtype=torch.long, device=self.device)
            old_log_probs = torch.tensor(
                np.array(buf["log_probs"]), dtype=torch.float32, device=self.device
            )
            rewards = np.array(buf["rewards"])
            dones = np.array(buf["dones"])
            values = np.array(buf["values"])

            # Compute GAE Advantages
            advantages = np.zeros_like(rewards, dtype=np.float32)
            last_gae = 0.0
            num_steps = len(rewards)

            for t in reversed(range(num_steps)):
                next_val = values[t + 1] if t + 1 < num_steps else 0.0
                next_non_terminal = 1.0 - float(dones[t])
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

            returns = advantages + values

            adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            if adv_t.numel() > 1 and float(adv_t.std().item()) > 1e-8:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            else:
                adv_t = adv_t - adv_t.mean()

            agent = self.agents[aid]
            optimizer = self.optimizers[aid]
            agent.train()

            for _ in range(num_epochs):
                _, new_lp, entropy, new_val = agent.get_action_and_value(obs, actions)

                ratio = torch.exp(new_lp - old_log_probs)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(new_val, ret_t)
                entropy_loss = -entropy.mean()

                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

                losses["policy_loss"] += policy_loss.item()
                losses["value_loss"] += value_loss.item()
                losses["entropy_loss"] += entropy_loss.item()
                num_updates += 1

        self.reset_buffers()
        if num_updates > 0:
            for k in losses:
                losses[k] /= num_updates
        return losses
