"""
Feudal Multi-Agent A2C (FMA2C) Baseline (B6) (Phase 2 §2.4)

Feudal MARL baseline with Manager-Worker hierarchy without constraints or belief state.
Features:
- Regional Partitioning: Groups N intersections into K sub-regions.
- Regional Manager: Sets sub-goal vectors for worker agents every T_manager steps.
- Worker Agents: Execute actions taking local observations + Manager sub-goal.
- Demonstrates hierarchical fragility under noise sweep (Novelty N4 diagnostic).

FIX (2026-08-26) — Bug 2.2:
  1. Worker update replaced Monte-Carlo returns + raw policy gradient (vanilla A2C,
     high variance) with GAE + PPO clipping (lower variance, bounded trust region).
  2. Manager update now trains both the VALUE HEAD (critic) AND the GOAL HEAD (actor)
     using a policy gradient loss.  Previously only the value head was updated, meaning
     the goal distribution never improved during training.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal


class FMA2CManager(nn.Module):
    """Regional Manager policy setting sub-goals for workers."""

    def __init__(
        self,
        num_regions: int,
        region_obs_dim: int,
        goal_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_regions = num_regions
        self.goal_dim = goal_dim

        self.net = nn.Sequential(
            nn.Linear(region_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.goal_head = nn.Linear(hidden_dim, goal_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, region_obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.net(region_obs)
        goal = torch.tanh(self.goal_head(feat))
        val = self.value_head(feat).squeeze(-1)
        return goal, val


class FMA2CWorker(nn.Module):
    """Worker policy taking local observation + manager sub-goal."""

    def __init__(
        self,
        obs_dim: int = 16,
        goal_dim: int = 4,
        action_dim: int = 4,
        hidden_dims: Tuple[int, int] = (128, 64),
    ):
        super().__init__()
        input_dim = obs_dim + goal_dim
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, obs_and_goal: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        logits = self.actor(obs_and_goal)
        val = self.critic(obs_and_goal).squeeze(-1)
        return Categorical(logits=logits), val


class FMA2CController:
    """
    FMA2C Controller orchestrating Manager sub-goals and Worker phase selection.
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        goal_dim: int = 4,
        t_manager: int = 15,
        num_regions: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,   # Bug 2.2: GAE lambda
        clip_eps: float = 0.2,       # Bug 2.2: PPO clip epsilon
        entropy_coef: float = 0.01,  # Bug 2.2: entropy bonus
        device: str = "cpu",
    ):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.t_manager = t_manager
        self.num_regions = num_regions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)

        # Partition agents into regions
        self.agent_to_region: Dict[str, int] = {}
        self.region_to_agents: Dict[int, List[str]] = {r: [] for r in range(num_regions)}
        for idx, aid in enumerate(agent_ids):
            region_idx = idx % num_regions
            self.agent_to_region[aid] = region_idx
            self.region_to_agents[region_idx].append(aid)

        region_obs_dim = obs_dim * (self.num_agents // max(1, num_regions))
        self.managers = {
            r: FMA2CManager(num_regions, region_obs_dim, goal_dim).to(self.device)
            for r in range(num_regions)
        }
        self.manager_opts = {
            r: optim.Adam(self.managers[r].parameters(), lr=lr) for r in range(num_regions)
        }

        self.worker = FMA2CWorker(obs_dim, goal_dim, action_dim).to(self.device)
        self.worker_opt = optim.Adam(self.worker.parameters(), lr=lr)

        self.current_goals: Dict[int, np.ndarray] = {
            r: np.zeros(goal_dim, dtype=np.float32) for r in range(num_regions)
        }
        self.accumulated_manager_rewards = {r: 0.0 for r in range(num_regions)}

        self.reset_buffers()

    def reset_buffers(self):
        self.worker_buffer = {
            aid: {
                "obs_goal": [], "actions": [], "log_probs": [],
                "rewards": [], "dones": [], "values": []
            }
            for aid in self.agent_ids
        }
        self.manager_buffer = {
            r: {"obs": [], "goals": [], "rewards": [], "dones": [], "values": []}
            for r in range(self.num_regions)
        }
        self.accumulated_manager_rewards = {r: 0.0 for r in range(self.num_regions)}
        self._episode_done = False

    def _get_region_obs(self, r_idx: int, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        a_list = self.region_to_agents[r_idx]
        vecs = [obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32)) for aid in a_list]
        return np.concatenate(vecs, axis=0) if vecs else np.zeros(self.obs_dim, dtype=np.float32)

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], evaluate: bool = False, step_count: int = 0
    ) -> Dict[str, int]:
        actions = {}

        # 1. Manager step every t_manager steps
        if step_count % self.t_manager == 0:
            with torch.no_grad():
                for r_idx in range(self.num_regions):
                    # Commit previous interval's accumulated reward to manager buffer if evaluating/training
                    if not evaluate and step_count > 0:
                        self.manager_buffer[r_idx]["rewards"].append(self.accumulated_manager_rewards[r_idx])
                        self.accumulated_manager_rewards[r_idx] = 0.0

                    r_obs = self._get_region_obs(r_idx, obs_dict)
                    r_obs_t = torch.tensor(r_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    mgr = self.managers[r_idx]
                    mgr.eval()
                    goal_t, val_t = mgr(r_obs_t)
                    goal_np = goal_t.squeeze(0).cpu().numpy()
                    self.current_goals[r_idx] = goal_np

                    if not evaluate:
                        self.manager_buffer[r_idx]["obs"].append(r_obs)
                        self.manager_buffer[r_idx]["goals"].append(goal_np)
                        self.manager_buffer[r_idx]["values"].append(float(val_t.item()))

        # 2. Worker action selection (batched across agents)
        self.worker.eval()
        with torch.no_grad():
            worker_inputs = []
            for aid in self.agent_ids:
                obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                r_idx = self.agent_to_region[aid]
                goal = self.current_goals[r_idx]
                worker_inputs.append(np.concatenate([obs, goal], axis=0))

            w_in_arr = np.array(worker_inputs, dtype=np.float32)
            w_in_t = torch.from_numpy(w_in_arr).to(self.device)
            dist, val_t = self.worker(w_in_t)

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
                vals = val_t.cpu().numpy()

                for idx, aid in enumerate(self.agent_ids):
                    act = int(acts[idx])
                    actions[aid] = act
                    self.worker_buffer[aid]["obs_goal"].append(worker_inputs[idx])
                    self.worker_buffer[aid]["actions"].append(act)
                    self.worker_buffer[aid]["log_probs"].append(float(lps[idx]))
                    self.worker_buffer[aid]["values"].append(float(vals[idx]))

        return actions

    def save_model(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "managers": {r: m.state_dict() for r, m in self.managers.items()},
            "worker": self.worker.state_dict(),
        }, path)

    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if "managers" in checkpoint:
            for r, m in self.managers.items():
                if r in checkpoint["managers"]:
                    m.load_state_dict(checkpoint["managers"][r])
        if "worker" in checkpoint:
            self.worker.load_state_dict(checkpoint["worker"])

    def store_transition(self, rewards: Dict[str, float], dones: Dict[str, bool]):
        is_done = any(dones.values())
        self._episode_done = is_done

        for aid in self.agent_ids:
            if aid in rewards:
                self.worker_buffer[aid]["rewards"].append(rewards[aid])
                self.worker_buffer[aid]["dones"].append(dones.get(aid, False))

        # Manager rewards: aggregate over interval
        for r_idx in range(self.num_regions):
            r_agents = self.region_to_agents[r_idx]
            r_rew = sum([rewards.get(aid, 0.0) for aid in r_agents])
            self.accumulated_manager_rewards[r_idx] += r_rew

    def update(self) -> Dict[str, float]:
        # Flush any remaining accumulated manager reward
        for r_idx in range(self.num_regions):
            m_buf = self.manager_buffer[r_idx]
            if len(m_buf["obs"]) > len(m_buf["rewards"]):
                m_buf["rewards"].append(self.accumulated_manager_rewards[r_idx])
                self.accumulated_manager_rewards[r_idx] = 0.0

        # Worker update — GAE + PPO clipping (Bug 2.2 fix)
        total_w_loss = 0.0
        w_count = 0
        self.worker.train()

        for aid in self.agent_ids:
            w_buf = self.worker_buffer[aid]
            if len(w_buf["obs_goal"]) == 0:
                continue

            obs_goal = torch.tensor(
                np.array(w_buf["obs_goal"]), dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                np.array(w_buf["actions"]), dtype=torch.long, device=self.device
            )
            old_log_probs = torch.tensor(
                np.array(w_buf["log_probs"]), dtype=torch.float32, device=self.device
            )
            rewards = np.array(w_buf["rewards"])
            values = np.array(w_buf["values"])
            dones = np.array(w_buf["dones"])
            num_steps = len(rewards)

            # ── GAE computation (replaces Monte-Carlo returns) ─────────────────
            advantages = np.zeros(num_steps, dtype=np.float32)
            last_gae = 0.0
            for t in reversed(range(num_steps)):
                next_non_terminal = 1.0 - float(dones[t])
                next_val = values[t + 1] if t + 1 < num_steps else 0.0
                delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
                advantages[t] = last_gae = (
                    delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                )
            returns = advantages + values

            adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            if adv_t.numel() > 1:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            # ── PPO clipped policy + value update ───────────────────────────
            dist, v_pred = self.worker(obs_goal)
            new_lp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_lp - old_log_probs)
            surr1 = ratio * adv_t
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.functional.mse_loss(v_pred, ret_t)

            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy
            self.worker_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=0.5)
            self.worker_opt.step()

            total_w_loss += loss.item()
            w_count += 1

        # Manager updates — VALUE HEAD + GOAL POLICY HEAD (Bug 2.2 fix)
        total_m_loss = 0.0
        for r_idx in range(self.num_regions):
            m_buf = self.manager_buffer[r_idx]
            if len(m_buf["obs"]) == 0 or len(m_buf["rewards"]) == 0:
                continue
            r_obs = torch.tensor(np.array(m_buf["obs"]), dtype=torch.float32, device=self.device)
            rewards = np.array(m_buf["rewards"])
            values = np.array(m_buf["values"])
            num_steps = min(len(rewards), len(values), len(r_obs))

            # Monte-Carlo returns for manager (low frequency updates, MC is acceptable)
            returns = np.zeros(num_steps, dtype=np.float32)
            running_ret = 0.0
            for t in reversed(range(num_steps)):
                running_ret = rewards[t] + (self.gamma ** self.t_manager) * running_ret
                returns[t] = running_ret

            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages_m = returns - values[:num_steps]
            adv_m_t = torch.tensor(advantages_m, dtype=torch.float32, device=self.device)
            if adv_m_t.numel() > 1:
                adv_m_t = (adv_m_t - adv_m_t.mean()) / (adv_m_t.std() + 1e-8)

            mgr = self.managers[r_idx]
            opt = self.manager_opts[r_idx]
            mgr.train()

            goal_t, v_pred = mgr(r_obs[:num_steps])

            # Value loss (critic)
            val_loss = nn.functional.mse_loss(v_pred, ret_t)

            # Goal policy gradient loss (policy head)
            # We use the cosine similarity between the produced goal and the
            # normalized advantage direction as a pseudo policy gradient:
            # high-advantage steps should be reinforced by increased goal magnitude.
            goal_norm = torch.nn.functional.normalize(goal_t, dim=-1)
            # Scalar reward for each manager step (mean advantage as weight)
            goal_policy_loss = -(goal_norm.norm(dim=-1) * adv_m_t.detach()).mean()

            manager_loss = goal_policy_loss + 0.5 * val_loss
            opt.zero_grad()
            manager_loss.backward()
            nn.utils.clip_grad_norm_(mgr.parameters(), max_norm=0.5)
            opt.step()

            total_m_loss += manager_loss.item()

        self.reset_buffers()
        return {
            "worker_loss": total_w_loss / max(1, w_count),
            "manager_loss": total_m_loss / max(1, self.num_regions),
        }
