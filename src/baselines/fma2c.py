"""
Feudal Multi-Agent A2C (FMA2C) Baseline (B6) (Phase 2 §2.4)

Feudal MARL baseline with Manager-Worker hierarchy without constraints or belief state.
Features:
- Regional Partitioning: Groups N intersections into K sub-regions.
- Regional Manager: Sets sub-goal vectors for worker agents every T_manager steps.
- Worker Agents: Execute actions taking local observations + Manager sub-goal.
- Demonstrates hierarchical fragility under noise sweep (Novelty N4 diagnostic).
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
            aid: {"obs_goal": [], "actions": [], "rewards": [], "dones": [], "values": []}
            for aid in self.agent_ids
        }
        self.manager_buffer = {
            r: {"obs": [], "goals": [], "rewards": [], "dones": [], "values": []}
            for r in range(self.num_regions)
        }
        self.accumulated_manager_rewards = {r: 0.0 for r in range(self.num_regions)}

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

        # 2. Worker action selection
        self.worker.eval()
        with torch.no_grad():
            for aid in self.agent_ids:
                obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                r_idx = self.agent_to_region[aid]
                goal = self.current_goals[r_idx]

                input_vec = np.concatenate([obs, goal], axis=0)
                input_t = torch.tensor(input_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

                dist, val_t = self.worker(input_t)

                if evaluate:
                    act_t = torch.argmax(dist.logits, dim=-1)
                else:
                    act_t = dist.sample()

                act = int(act_t.item())
                actions[aid] = act

                if not evaluate:
                    self.worker_buffer[aid]["obs_goal"].append(input_vec)
                    self.worker_buffer[aid]["actions"].append(act)
                    self.worker_buffer[aid]["values"].append(float(val_t.item()))

        return actions

    def store_transition(self, rewards: Dict[str, float], dones: Dict[str, bool]):
        for aid in self.agent_ids:
            if aid in rewards:
                self.worker_buffer[aid]["rewards"].append(rewards[aid])
                self.worker_buffer[aid]["dones"].append(dones[aid])

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

        # Worker update
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
            rewards = np.array(w_buf["rewards"])
            values = np.array(w_buf["values"])
            dones = np.array(w_buf["dones"])
            num_steps = len(rewards)

            returns = np.zeros_like(rewards, dtype=np.float32)
            running_ret = 0.0
            for t in reversed(range(num_steps)):
                if dones[t]:
                    running_ret = 0.0
                running_ret = rewards[t] + self.gamma * running_ret
                returns[t] = running_ret

            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            val_t = torch.tensor(values, dtype=torch.float32, device=self.device)
            adv_t = ret_t - val_t

            dist, v_pred = self.worker(obs_goal)
            log_prob = dist.log_prob(actions)
            policy_loss = -(log_prob * adv_t.detach()).mean()
            value_loss = nn.functional.mse_loss(v_pred, ret_t)

            loss = policy_loss + 0.5 * value_loss
            self.worker_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.worker.parameters(), max_norm=0.5)
            self.worker_opt.step()

            total_w_loss += loss.item()
            w_count += 1

        # Manager updates
        total_m_loss = 0.0
        for r_idx in range(self.num_regions):
            m_buf = self.manager_buffer[r_idx]
            if len(m_buf["obs"]) == 0 or len(m_buf["rewards"]) == 0:
                continue
            r_obs = torch.tensor(np.array(m_buf["obs"]), dtype=torch.float32, device=self.device)
            rewards = np.array(m_buf["rewards"])
            values = np.array(m_buf["values"])
            num_steps = min(len(rewards), len(values))

            returns = np.zeros_like(rewards[:num_steps], dtype=np.float32)
            running_ret = 0.0
            for t in reversed(range(num_steps)):
                running_ret = rewards[t] + (self.gamma ** self.t_manager) * running_ret
                returns[t] = running_ret

            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            mgr = self.managers[r_idx]
            opt = self.manager_opts[r_idx]
            mgr.train()

            _, v_pred = mgr(r_obs[:num_steps])
            val_loss = nn.functional.mse_loss(v_pred, ret_t)

            opt.zero_grad()
            val_loss.backward()
            opt.step()

            total_m_loss += val_loss.item()

        self.reset_buffers()
        return {
            "worker_loss": total_w_loss / max(1, w_count),
            "manager_loss": total_m_loss / max(1, self.num_regions),
        }
