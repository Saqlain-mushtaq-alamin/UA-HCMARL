"""
Multi-Agent PPO (MAPPO) Baseline (B4) (Phase 2 §2.2)

Multi-Agent PPO with Centralized Critic.
Features:
- Parameter-shared or individual Actor networks operating on local observations.
- Centralized Critic network operating on concatenated global state vector.
- Centralized GAE advantage estimation.

FIX (2026-08-26) — Bug 2.1:
  When share_actor=True the shared network was updated once per agent per epoch
  (16 agents × 4 epochs = 64 gradient steps) using progressively staler old_log_probs.
  Fix: when shared, all agent observations are concatenated into one batch and the
  network receives a single set of epoch updates with consistent old_log_probs.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class MAPPOActor(nn.Module):
    """Local observation policy network for MAPPO."""

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        hidden_dims: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
        )

    def forward(self, obs: torch.Tensor) -> Categorical:
        logits = self.net(obs)
        return Categorical(logits=logits)


class MAPPOCentralizedCritic(nn.Module):
    """Centralized Value network taking concatenated global observations."""

    def __init__(
        self,
        global_state_dim: int,
        hidden_dims: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class MAPPOController:
    """
    MAPPO Controller managing centralized critic and actor networks.
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        share_actor: bool = True,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        device: str = "cpu",
    ):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.global_state_dim = obs_dim * self.num_agents
        self.share_actor = share_actor
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        # Centralized Critic
        self.critic = MAPPOCentralizedCritic(self.global_state_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        # Actors
        if share_actor:
            self.shared_actor = MAPPOActor(obs_dim, action_dim).to(self.device)
            self.actor_optimizer = optim.Adam(self.shared_actor.parameters(), lr=lr)
        else:
            self.actors = {
                aid: MAPPOActor(obs_dim, action_dim).to(self.device) for aid in agent_ids
            }
            self.actor_optimizers = {
                aid: optim.Adam(self.actors[aid].parameters(), lr=lr) for aid in agent_ids
            }

        self.reset_buffers()

    def reset_buffers(self):
        self.buffer = {
            "obs": {aid: [] for aid in self.agent_ids},
            "global_states": [],
            "actions": {aid: [] for aid in self.agent_ids},
            "log_probs": {aid: [] for aid in self.agent_ids},
            "rewards": [],
            "dones": [],
            "values": [],
        }

    def _get_global_state(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        vecs = [obs_dict[aid] for aid in self.agent_ids if aid in obs_dict]
        if len(vecs) < self.num_agents:
            # fill missing
            padding = [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.num_agents - len(vecs))]
            vecs.extend(padding)
        return np.concatenate(vecs, axis=0)

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], evaluate: bool = False, step_count: int = 0
    ) -> Dict[str, int]:
        actions = {}
        global_state = self._get_global_state(obs_dict)

        with torch.no_grad():
            gs_t = torch.tensor(global_state, dtype=torch.float32, device=self.device).unsqueeze(0)
            value = float(self.critic(gs_t).item())

            if self.share_actor:
                obs_list = [obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32)) for aid in self.agent_ids]
                obs_arr = np.array(obs_list, dtype=np.float32)
                obs_t = torch.from_numpy(obs_arr).to(self.device)
                self.shared_actor.eval()
                dist = self.shared_actor(obs_t)

                if evaluate:
                    act_t = torch.argmax(dist.logits, dim=-1)
                    acts = act_t.cpu().numpy()
                    for idx, aid in enumerate(self.agent_ids):
                        actions[aid] = int(acts[idx])
                else:
                    act_t = dist.sample()
                    log_prob_t = dist.log_prob(act_t)
                    acts = act_t.cpu().numpy()
                    lps = log_prob_t.cpu().numpy()
                    for idx, aid in enumerate(self.agent_ids):
                        act = int(acts[idx])
                        actions[aid] = act
                        self.buffer["obs"][aid].append(obs_list[idx])
                        self.buffer["actions"][aid].append(act)
                        self.buffer["log_probs"][aid].append(float(lps[idx]))
            else:
                for aid in self.agent_ids:
                    obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                    obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)

                    actor = self.actors[aid]
                    actor.eval()
                    dist = actor(obs_t)

                    if evaluate:
                        act_t = torch.argmax(dist.logits, dim=-1)
                        log_prob_t = torch.zeros(1, device=self.device)
                    else:
                        act_t = dist.sample()
                        log_prob_t = dist.log_prob(act_t)

                    act = int(act_t.item())
                    actions[aid] = act

                    if not evaluate:
                        self.buffer["obs"][aid].append(obs)
                        self.buffer["actions"][aid].append(act)
                        self.buffer["log_probs"][aid].append(float(log_prob_t.item()))

            if not evaluate:
                self.buffer["global_states"].append(global_state)
                self.buffer["values"].append(value)

        return actions

    def save_model(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "share_actor": self.share_actor,
            "shared_actor": self.shared_actor.state_dict() if self.share_actor else None,
            "actors": {aid: a.state_dict() for aid, a in self.actors.items()} if not self.share_actor else None,
            "critic": self.critic.state_dict(),
        }
        torch.save(data, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if self.share_actor and checkpoint.get("shared_actor") is not None:
            self.shared_actor.load_state_dict(checkpoint["shared_actor"])
        elif not self.share_actor and checkpoint.get("actors") is not None:
            for aid, a in self.actors.items():
                if aid in checkpoint["actors"]:
                    a.load_state_dict(checkpoint["actors"][aid])
        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])

    def store_transition(self, rewards: Dict[str, float], dones: Dict[str, bool]):
        team_reward = float(np.mean([rewards[aid] for aid in self.agent_ids if aid in rewards]))
        is_done = any(dones.values())
        self.buffer["rewards"].append(team_reward)
        self.buffer["dones"].append(is_done)

    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        if len(self.buffer["global_states"]) == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}

        global_states = torch.tensor(
            np.array(self.buffer["global_states"]), dtype=torch.float32, device=self.device
        )
        rewards = np.array(self.buffer["rewards"])
        dones = np.array(self.buffer["dones"])
        values = np.array(self.buffer["values"])
        num_steps = len(rewards)

        # Compute GAE on Centralized Critic
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
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

        # 1. Update Centralized Critic
        self.critic.train()
        for _ in range(num_epochs):
            v_pred = self.critic(global_states)
            val_loss = nn.functional.mse_loss(v_pred, ret_t)
            self.critic_optimizer.zero_grad()
            val_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
            self.critic_optimizer.step()

        # 2. Update Actor(s)
        total_p_loss = 0.0
        total_ent_loss = 0.0
        update_count = 0

        if self.share_actor:
            # ── Bug 2.1 fix: aggregate ALL agents into ONE batch ─────────────────
            # Concatenate observations, actions, and log-probs across agents.
            # The advantage tensor (step-level) is repeated once per agent entry
            # so that indices line up correctly.
            all_obs = torch.cat(
                [torch.tensor(np.array(self.buffer["obs"][aid]),
                              dtype=torch.float32, device=self.device)
                 for aid in self.agent_ids],
                dim=0,
            )
            all_actions = torch.cat(
                [torch.tensor(np.array(self.buffer["actions"][aid]),
                              dtype=torch.long, device=self.device)
                 for aid in self.agent_ids],
                dim=0,
            )
            all_old_lp = torch.cat(
                [torch.tensor(np.array(self.buffer["log_probs"][aid]),
                              dtype=torch.float32, device=self.device)
                 for aid in self.agent_ids],
                dim=0,
            )
            # Repeat the step-level advantages once per agent
            all_adv = adv_t.repeat(self.num_agents)

            actor = self.shared_actor
            opt = self.actor_optimizer
            actor.train()
            for _ in range(num_epochs):
                dist = actor(all_obs)
                new_lp = dist.log_prob(all_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - all_old_lp)
                surr1 = ratio * all_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * all_adv
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                opt.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                opt.step()

                total_p_loss += policy_loss.item()
                total_ent_loss += entropy.item()
                update_count += 1
        else:
            # Individual actors: one batch per agent, same num_epochs each
            for aid in self.agent_ids:
                obs = torch.tensor(
                    np.array(self.buffer["obs"][aid]), dtype=torch.float32, device=self.device
                )
                actions = torch.tensor(
                    np.array(self.buffer["actions"][aid]), dtype=torch.long, device=self.device
                )
                old_log_probs = torch.tensor(
                    np.array(self.buffer["log_probs"][aid]), dtype=torch.float32, device=self.device
                )

                actor = self.actors[aid]
                opt = self.actor_optimizers[aid]
                actor.train()

                for _ in range(num_epochs):
                    dist = actor(obs)
                    new_lp = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - old_log_probs)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                    opt.zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    opt.step()

                    total_p_loss += policy_loss.item()
                    total_ent_loss += entropy.item()
                    update_count += 1

        self.reset_buffers()
        return {
            "policy_loss": total_p_loss / max(1, update_count),
            "value_loss": val_loss.item(),
            "entropy_loss": total_ent_loss / max(1, update_count),
        }
