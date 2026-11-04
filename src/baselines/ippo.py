"""
Independent PPO Baseline (B3) (Phase 2 §2.2)

Independent PPO per intersection with optional parameter sharing and action masking.
Features:
- Parameter sharing option (share_actor=True, share_critic=True, default) across intersections
  with decentralized local observation critic (not centralized global state).
- Action masking support to prevent invalid transitions during min-green and clearance intervals.
- Standard PPO policy clip and value loss updates.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class IPPOActor(nn.Module):
    def __init__(self, obs_dim: int = 16, action_dim: int = 4, hidden_dims: Tuple[int, int] = (128, 64)):
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


class IPPOCritic(nn.Module):
    def __init__(self, obs_dim: int = 16, hidden_dims: Tuple[int, int] = (128, 64)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class IPPOAgent(nn.Module):
    """Single agent PPO actor-critic model (for unshared mode)."""

    def __init__(
        self,
        obs_dim: int = 16,
        action_dim: int = 4,
        hidden_dims: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        self.actor = IPPOActor(obs_dim, action_dim, hidden_dims)
        self.critic = IPPOCritic(obs_dim, hidden_dims)

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor.net(obs)
        if mask is not None:
            logits = torch.where(mask > 0, logits, torch.tensor(-1e9, device=logits.device))
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(obs)
        return action, log_prob, entropy, value


class IPPOController:
    """
    Multi-Agent Independent PPO controller wrapper.
    Supports parameter sharing (default, standard MARL benchmark) and action masking.
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        share_actor: bool = True,
        share_critic: bool = True,
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
        self.share_actor = share_actor
        self.share_critic = share_critic
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.device = torch.device(device)

        if self.share_actor:
            self.shared_actor = IPPOActor(obs_dim, action_dim).to(self.device)
            self.actor_optimizer = optim.Adam(self.shared_actor.parameters(), lr=lr)
        else:
            self.actors = {aid: IPPOActor(obs_dim, action_dim).to(self.device) for aid in agent_ids}
            self.actor_optimizers = {aid: optim.Adam(self.actors[aid].parameters(), lr=lr) for aid in agent_ids}

        if self.share_critic:
            self.shared_critic = IPPOCritic(obs_dim).to(self.device)
            self.critic_optimizer = optim.Adam(self.shared_critic.parameters(), lr=lr)
        else:
            self.critics = {aid: IPPOCritic(obs_dim).to(self.device) for aid in agent_ids}
            self.critic_optimizers = {aid: optim.Adam(self.critics[aid].parameters(), lr=lr) for aid in agent_ids}

        # Backwards compatibility agents map
        self.agents: Dict[str, Any] = {}

        self.reset_buffers()

    def reset(self, agent_ids: Optional[List[str]] = None):
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

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        evaluate: bool = False,
        step_count: int = 0,
        action_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, int]:
        actions = {}
        with torch.no_grad():
            if self.share_actor and self.share_critic:
                obs_list = [obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32)) for aid in self.agent_ids]
                obs_arr = np.array(obs_list, dtype=np.float32)
                obs_t = torch.from_numpy(obs_arr).to(self.device)

                self.shared_actor.eval()
                self.shared_critic.eval()

                logits = self.shared_actor.net(obs_t)
                if action_masks is not None:
                    masks_list = [
                        action_masks.get(aid, np.ones(self.action_dim, dtype=np.float32))
                        for aid in self.agent_ids
                    ]
                    mask_t = torch.tensor(np.array(masks_list), dtype=torch.float32, device=self.device)
                    logits = torch.where(mask_t > 0.0, logits, torch.tensor(-1e9, device=self.device))

                vals_t = self.shared_critic(obs_t)
                vals = vals_t.cpu().numpy()

                if evaluate:
                    act_t = torch.argmax(logits, dim=-1)
                    acts = act_t.cpu().numpy()
                    for idx, aid in enumerate(self.agent_ids):
                        actions[aid] = int(acts[idx])
                else:
                    dist = Categorical(logits=logits)
                    act_t = dist.sample()
                    lp_t = dist.log_prob(act_t)
                    acts = act_t.cpu().numpy()
                    lps = lp_t.cpu().numpy()

                    for idx, aid in enumerate(self.agent_ids):
                        act = int(acts[idx])
                        actions[aid] = act
                        self.buffers[aid]["obs"].append(obs_list[idx])
                        self.buffers[aid]["actions"].append(act)
                        self.buffers[aid]["log_probs"].append(float(lps[idx]))
                        self.buffers[aid]["values"].append(float(vals[idx]))
            else:
                for aid in self.agent_ids:
                    if aid in obs_dict:
                        obs = obs_dict[aid]
                        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        actor = self.shared_actor if self.share_actor else self.actors[aid]
                        critic = self.shared_critic if self.share_critic else self.critics[aid]
                        actor.eval()
                        critic.eval()

                        logits = actor.net(obs_t)
                        if action_masks is not None and aid in action_masks:
                            mask_t = torch.tensor(action_masks[aid], dtype=torch.float32, device=self.device).unsqueeze(0)
                            logits = torch.where(mask_t > 0.0, logits, torch.tensor(-1e9, device=self.device))

                        val_t = critic(obs_t)

                        if evaluate:
                            act_t = torch.argmax(logits, dim=-1)
                            actions[aid] = int(act_t.item())
                        else:
                            dist = Categorical(logits=logits)
                            act_t = dist.sample()
                            lp_t = dist.log_prob(act_t)
                            act = int(act_t.item())
                            actions[aid] = act
                            self.buffers[aid]["obs"].append(obs)
                            self.buffers[aid]["actions"].append(act)
                            self.buffers[aid]["log_probs"].append(float(lp_t.item()))
                            self.buffers[aid]["values"].append(float(val_t.item()))
                    else:
                        actions[aid] = 0
        return actions

    def store_transition(self, rewards: Dict[str, float], dones: Dict[str, bool]):
        for aid in self.agent_ids:
            if aid in rewards:
                self.buffers[aid]["rewards"].append(rewards[aid])
                self.buffers[aid]["dones"].append(dones[aid])

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "share_actor": self.share_actor,
            "share_critic": self.share_critic,
            "shared_actor": self.shared_actor.state_dict() if self.share_actor else None,
            "shared_critic": self.shared_critic.state_dict() if self.share_critic else None,
            "actors": {aid: a.state_dict() for aid, a in self.actors.items()} if not self.share_actor else None,
            "critics": {aid: c.state_dict() for aid, c in self.critics.items()} if not self.share_critic else None,
        }
        torch.save(data, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        # Check if legacy checkpoint format {aid: state_dict}
        if isinstance(checkpoint, dict) and "shared_actor" not in checkpoint and "actors" not in checkpoint:
            # Legacy independent model format
            if self.share_actor and len(checkpoint) > 0:
                first_key = list(checkpoint.keys())[0]
                first_state = checkpoint[first_key]
                # Extract actor net weights if present
                actor_dict = {k.replace("actor.net.", "").replace("actor.", ""): v for k, v in first_state.items() if "actor" in k}
                if len(actor_dict) > 0:
                    try:
                        self.shared_actor.net.load_state_dict(actor_dict)
                    except Exception:
                        pass
            return

        if self.share_actor and checkpoint.get("shared_actor") is not None:
            self.shared_actor.load_state_dict(checkpoint["shared_actor"])
        elif not self.share_actor and checkpoint.get("actors") is not None:
            for aid, a in self.actors.items():
                if aid in checkpoint["actors"]:
                    a.load_state_dict(checkpoint["actors"][aid])

        if self.share_critic and checkpoint.get("shared_critic") is not None:
            self.shared_critic.load_state_dict(checkpoint["shared_critic"])
        elif not self.share_critic and checkpoint.get("critics") is not None:
            for aid, c in self.critics.items():
                if aid in checkpoint["critics"]:
                    c.load_state_dict(checkpoint["critics"][aid])

    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        if self.share_actor and self.share_critic:
            all_obs = []
            all_acts = []
            all_old_lps = []
            all_advs = []
            all_rets = []

            for aid in self.agent_ids:
                buf = self.buffers[aid]
                if len(buf["obs"]) == 0:
                    continue
                obs = np.array(buf["obs"])
                actions = np.array(buf["actions"])
                old_lp = np.array(buf["log_probs"])
                rewards = np.array(buf["rewards"])
                dones = np.array(buf["dones"])
                values = np.array(buf["values"])

                num_steps = len(rewards)
                advantages = np.zeros_like(rewards, dtype=np.float32)
                last_gae = 0.0
                for t in reversed(range(num_steps)):
                    next_val = values[t + 1] if t + 1 < num_steps else 0.0
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
                return {"policy_loss": 0.0, "value_loss": 0.0, "entropy_loss": 0.0}

            obs_t = torch.tensor(np.concatenate(all_obs, axis=0), dtype=torch.float32, device=self.device)
            acts_t = torch.tensor(np.concatenate(all_acts, axis=0), dtype=torch.long, device=self.device)
            old_lp_t = torch.tensor(np.concatenate(all_old_lps, axis=0), dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(np.concatenate(all_advs, axis=0), dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(np.concatenate(all_rets, axis=0), dtype=torch.float32, device=self.device)

            if adv_t.numel() > 1 and float(adv_t.std().item()) > 1e-8:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
            else:
                adv_t = adv_t - adv_t.mean()

            self.shared_actor.train()
            self.shared_critic.train()

            total_p_loss = 0.0
            total_v_loss = 0.0
            total_ent = 0.0

            for _ in range(num_epochs):
                dist = self.shared_actor(obs_t)
                new_lp = dist.log_prob(acts_t)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_lp - old_lp_t)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                self.actor_optimizer.zero_grad()
                policy_loss.backward()
                nn.utils.clip_grad_norm_(self.shared_actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                v_pred = self.shared_critic(obs_t)
                value_loss = nn.functional.mse_loss(v_pred, ret_t)
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.shared_critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                total_p_loss += policy_loss.item()
                total_v_loss += value_loss.item()
                total_ent += entropy.item()

            self.reset_buffers()
            return {
                "policy_loss": total_p_loss / num_epochs,
                "value_loss": total_v_loss / num_epochs,
                "entropy_loss": total_ent / num_epochs,
            }
        else:
            # Independent unshared update
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

                actor = self.actors[aid]
                critic = self.critics[aid]
                actor.train()
                critic.train()

                for _ in range(num_epochs):
                    dist = actor(obs)
                    new_lp = dist.log_prob(actions)
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_lp - old_log_probs)
                    surr1 = ratio * adv_t
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
                    policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                    self.actor_optimizers[aid].zero_grad()
                    policy_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
                    self.actor_optimizers[aid].step()

                    v_pred = critic(obs)
                    value_loss = nn.functional.mse_loss(v_pred, ret_t)
                    self.critic_optimizers[aid].zero_grad()
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
                    self.critic_optimizers[aid].step()

                    losses["policy_loss"] += policy_loss.item()
                    losses["value_loss"] += value_loss.item()
                    losses["entropy_loss"] += entropy.item()
                    num_updates += 1

            self.reset_buffers()
            if num_updates > 0:
                for k in losses:
                    losses[k] /= num_updates
            return losses
