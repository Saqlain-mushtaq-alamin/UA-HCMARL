"""
Stage 4 Full System: UA-HCMARL Controller (Phase 4 / Blueprint §6)

Complete Uncertainty-Aware Hierarchical Constrained Multi-Agent Reinforcement Learning System.
Combines:
- Layer 1: GRU-KalmanNet Belief State Encoder (b_hat_i, sigma_hat_i)
- Layer 2: Constrained MAPPO Workers with PID-Lagrangian multipliers
- Layer 3: LP-Projected Uncertainty-Aware Regional Manager Agents (acting every T_manager = 15 steps)
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.belief_encoder import GRUKalmanNet
from src.models.lagrangian import PIDLagrangianMultiplier
from src.models.manager_agent import UncertaintyAwareManager, compute_manager_state
from src.baselines.mappo import MAPPOActor, MAPPOCentralizedCritic


class UAHCMARLController:
    """
    Complete UA-HCMARL Controller (Layer 1 + Layer 2 + Layer 3).
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        history_len: int = 5,
        t_manager: int = 15,
        num_regions: int = 4,
        b_total_per_region: float = 100.0,
        lr_bse: float = 5e-5,
        lr_worker: float = 1e-4,
        lr_manager: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        freeze_bse: bool = False,
        device: str = "cpu",
    ):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.t_manager = t_manager
        self.num_regions = num_regions
        self.b_total_per_region = b_total_per_region
        self.freeze_bse = freeze_bse
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.device = torch.device(device)

        # Region Partitioning (Static 4x4 -> 4 regions of 4 workers each)
        eff_num_regions = min(num_regions, self.num_agents) if self.num_agents > 0 else 1
        self.num_workers_per_region = max(1, self.num_agents // eff_num_regions) if eff_num_regions > 0 else 1
        self.region_assignments: Dict[int, List[str]] = {}
        for r in range(eff_num_regions):
            start_idx = r * self.num_workers_per_region
            end_idx = start_idx + self.num_workers_per_region if r < eff_num_regions - 1 else self.num_agents
            self.region_assignments[r] = self.agent_ids[start_idx:end_idx]

        self.agent_to_region: Dict[str, int] = {}
        for r, aids in self.region_assignments.items():
            for aid in aids:
                self.agent_to_region[aid] = r

        # Layer 1: GRU-KalmanNet BSE
        self.bse = GRUKalmanNet(raw_obs_dim=obs_dim, hidden_dim=64, belief_dim=obs_dim, history_len=history_len).to(self.device)

        # Layer 2: Workers
        self.worker_obs_dim = obs_dim + 1  # b_hat + sigma_hat
        global_state_dim = self.worker_obs_dim * self.num_agents

        self.worker_actor = MAPPOActor(self.worker_obs_dim, action_dim).to(self.device)
        self.worker_r_critic = MAPPOCentralizedCritic(global_state_dim).to(self.device)
        self.worker_c_critic = MAPPOCentralizedCritic(global_state_dim).to(self.device)

        # PID Lagrangian per agent
        self.pid_lagrangians = {aid: PIDLagrangianMultiplier(use_confidence_modulation=True) for aid in agent_ids}

        # Layer 3: Regional Managers
        manager_state_dim = obs_dim + 5
        self.managers = {
            r: UncertaintyAwareManager(
                num_workers=len(aids),
                state_dim=manager_state_dim,
                hidden_dim=64,
                b_total=b_total_per_region,
            ).to(self.device)
            for r, aids in self.region_assignments.items()
        }

        # Optimizers
        self.bse_optimizer = optim.Adam(self.bse.parameters(), lr=lr_bse)
        self.worker_actor_optimizer = optim.Adam(self.worker_actor.parameters(), lr=lr_worker)
        self.worker_r_critic_optimizer = optim.Adam(self.worker_r_critic.parameters(), lr=lr_worker)
        self.worker_c_critic_optimizer = optim.Adam(self.worker_c_critic.parameters(), lr=lr_worker)

        self.manager_optimizers = {
            r: optim.Adam(mgr.parameters(), lr=lr_manager) for r, mgr in self.managers.items()
        }

        # Dynamic Assigned Budgets per worker
        default_b = b_total_per_region / self.num_workers_per_region
        self.worker_budgets: Dict[str, float] = {aid: default_b for aid in agent_ids}

        # Buffers
        self.obs_history_buffers: Dict[str, List[np.ndarray]] = {aid: [] for aid in agent_ids}
        self.prev_beliefs: Dict[str, torch.Tensor] = {}
        self.latest_worker_reports: Dict[str, Dict[str, Any]] = {}

        self.reset_buffers()

    def reset_buffers(self):
        self.obs_history_buffers = {aid: [] for aid in self.agent_ids}
        self.prev_beliefs = {}
        self.latest_worker_reports = {}

        default_b = self.b_total_per_region / self.num_workers_per_region
        self.worker_budgets = {aid: default_b for aid in self.agent_ids}

        self.worker_buffer = {
            "obs": {aid: [] for aid in self.agent_ids},
            "b_hat": {aid: [] for aid in self.agent_ids},
            "sigma_hat": {aid: [] for aid in self.agent_ids},
            "worker_input": {aid: [] for aid in self.agent_ids},
            "actions": {aid: [] for aid in self.agent_ids},
            "log_probs": {aid: [] for aid in self.agent_ids},
            "rewards": {aid: [] for aid in self.agent_ids},
            "costs": {aid: [] for aid in self.agent_ids},
            "dones": {aid: [] for aid in self.agent_ids},
            "global_states": [],
            "reward_values": [],
            "cost_values": [],
        }

        self.manager_buffer = {
            r: {"states": [], "budgets": [], "rewards": [], "log_probs": [], "values": []}
            for r in self.managers.keys()
        }

    def _get_history_tensor(self, aid: str, current_obs: np.ndarray) -> torch.Tensor:
        buf = self.obs_history_buffers[aid]
        buf.append(current_obs)
        if len(buf) > self.history_len:
            buf.pop(0)

        pad_len = self.history_len - len(buf)
        padded = [buf[0]] * pad_len + buf
        arr = np.array(padded, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], evaluate: bool = False, step_count: int = 0
    ) -> Dict[str, int]:
        actions = {}
        worker_input_dict = {}

        # 1. Multi-Timescale Manager Budget Allocation every T_manager steps
        if step_count % self.t_manager == 0 and step_count > 0:
            for r, aids in self.region_assignments.items():
                reports = [self.latest_worker_reports.get(aid, {}) for aid in aids]
                mgr_state = compute_manager_state(reports, total_budget=self.b_total_per_region)
                mgr_state_t = torch.tensor(mgr_state, dtype=torch.float32, device=self.device).unsqueeze(0)

                mgr = self.managers[r]
                mgr.eval()
                with torch.no_grad():
                    dist, budget_t, val_t = mgr(mgr_state_t)
                    budgets_np = budget_t.squeeze(0).cpu().numpy()

                for idx, aid in enumerate(aids):
                    self.worker_budgets[aid] = float(budgets_np[idx])

                if not evaluate:
                    self.manager_buffer[r]["states"].append(mgr_state)
                    self.manager_buffer[r]["budgets"].append(budgets_np)
                    self.manager_buffer[r]["values"].append(float(val_t.item()))

        # 2. Worker Action Selection
        self.bse.eval()
        self.worker_actor.eval()

        with torch.no_grad():
            for aid in self.agent_ids:
                raw_obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                hist_t = self._get_history_tensor(aid, raw_obs)
                prev_b = self.prev_beliefs.get(aid, None)

                b_hat_t, sigma_hat_t, _ = self.bse(hist_t, prev_belief=prev_b)
                self.prev_beliefs[aid] = b_hat_t.detach()

                b_hat_np = b_hat_t.squeeze(0).cpu().numpy()
                sigma_hat_val = float(sigma_hat_t.item())

                # Store latest report for upward communication
                self.latest_worker_reports[aid] = {
                    "belief": b_hat_np,
                    "sigma": sigma_hat_val,
                    "violation_rate": 0.0,
                }

                worker_input = np.concatenate([b_hat_np, [sigma_hat_val]], axis=0)
                worker_input_dict[aid] = worker_input
                worker_input_t = torch.tensor(worker_input, dtype=torch.float32, device=self.device).unsqueeze(0)

                dist = self.worker_actor(worker_input_t)
                if evaluate:
                    act_t = torch.argmax(dist.logits, dim=-1)
                else:
                    act_t = dist.sample()

                act = int(act_t.item())
                actions[aid] = act

                if not evaluate:
                    log_prob = float(dist.log_prob(act_t).item())
                    self.worker_buffer["obs"][aid].append(raw_obs)
                    self.worker_buffer["b_hat"][aid].append(b_hat_np)
                    self.worker_buffer["sigma_hat"][aid].append(sigma_hat_val)
                    self.worker_buffer["worker_input"][aid].append(worker_input)
                    self.worker_buffer["actions"][aid].append(act)
                    self.worker_buffer["log_probs"][aid].append(log_prob)

            if not evaluate:
                global_vec = np.concatenate([worker_input_dict[aid] for aid in self.agent_ids], axis=0)
                global_t = torch.tensor(global_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
                r_val = float(self.worker_r_critic(global_t).item())
                c_val = float(self.worker_c_critic(global_t).item())

                self.worker_buffer["global_states"].append(global_vec)
                self.worker_buffer["reward_values"].append(r_val)
                self.worker_buffer["cost_values"].append(c_val)

        return actions

    def store_transition(
        self,
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        step_violations: Optional[Dict[str, int]] = None,
    ):
        for aid in self.agent_ids:
            if aid in rewards:
                self.worker_buffer["rewards"][aid].append(rewards[aid])
                self.worker_buffer["dones"][aid].append(dones[aid])

        if step_violations:
            c_vec = np.array(
                [
                    step_violations.get("green_time", 0),
                    step_violations.get("green_skip", 0),
                    step_violations.get("phase_skip", 0),
                ],
                dtype=np.float32,
            )
            for aid in self.agent_ids:
                sigma_val = self.worker_buffer["sigma_hat"][aid][-1] if self.worker_buffer["sigma_hat"][aid] else 1.0
                budget_val = self.worker_budgets.get(aid, 25.0)

                lambdas = self.pid_lagrangians[aid].update(
                    constraint_costs=c_vec,
                    constraint_limits=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                    sigma_hat=sigma_val,
                    budget=budget_val,
                )
                cost_val = float(np.sum(c_vec * lambdas))
                self.worker_buffer["costs"][aid].append(cost_val)
        else:
            for aid in self.agent_ids:
                self.worker_buffer["costs"][aid].append(0.0)

    def update(self) -> Dict[str, float]:
        if len(self.worker_buffer["global_states"]) == 0:
            return {}

        # 1. Update Workers
        global_states = torch.tensor(np.array(self.worker_buffer["global_states"]), dtype=torch.float32, device=self.device)
        r_values = np.array(self.worker_buffer["reward_values"], dtype=np.float32)
        mean_rewards = np.mean([np.array(self.worker_buffer["rewards"][aid]) for aid in self.agent_ids], axis=0)
        mean_costs = np.mean([np.array(self.worker_buffer["costs"][aid]) for aid in self.agent_ids], axis=0)
        dones = np.array(self.worker_buffer["dones"][self.agent_ids[0]])

        net_rewards = mean_rewards - mean_costs
        num_steps = len(net_rewards)

        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(num_steps)):
            next_non_terminal = 1.0 - float(dones[t])
            next_val = r_values[t + 1] if t + 1 < num_steps else 0.0
            delta = net_rewards[t] + self.gamma * next_val * next_non_terminal - r_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + r_values
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if adv_t.numel() > 1 and float(adv_t.std().item()) > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        else:
            adv_t = adv_t - adv_t.mean()

        self.worker_r_critic.train()
        v_pred = self.worker_r_critic(global_states)
        loss_r_critic = nn.functional.mse_loss(v_pred, ret_t)

        self.worker_r_critic_optimizer.zero_grad()
        loss_r_critic.backward()
        self.worker_r_critic_optimizer.step()

        self.worker_actor.train()
        total_p_loss = 0.0
        for aid in self.agent_ids:
            w_input = torch.tensor(np.array(self.worker_buffer["worker_input"][aid]), dtype=torch.float32, device=self.device)
            actions = torch.tensor(np.array(self.worker_buffer["actions"][aid]), dtype=torch.long, device=self.device)
            old_lps = torch.tensor(np.array(self.worker_buffer["log_probs"][aid]), dtype=torch.float32, device=self.device)

            dist = self.worker_actor(w_input)
            new_lps = dist.log_prob(actions)
            ratios = torch.exp(new_lps - old_lps)

            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            self.worker_actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.worker_actor.parameters(), max_norm=0.5)
            self.worker_actor_optimizer.step()
            total_p_loss += policy_loss.item()

        # 2. Update Managers
        total_mgr_loss = 0.0
        for r, mgr in self.managers.items():
            if len(self.manager_buffer[r]["states"]) > 0:
                mgr_states = torch.tensor(np.array(self.manager_buffer[r]["states"]), dtype=torch.float32, device=self.device)
                mgr.train()
                dist, budget_t, val_t = mgr(mgr_states)
                mgr_loss = -val_t.mean()

                self.manager_optimizers[r].zero_grad()
                mgr_loss.backward()
                nn.utils.clip_grad_norm_(mgr.parameters(), max_norm=0.5)
                self.manager_optimizers[r].step()
                total_mgr_loss += float(mgr_loss.item())

        self.reset_buffers()
        return {
            "worker_policy_loss": total_p_loss / self.num_agents,
            "worker_value_loss": float(loss_r_critic.item()),
            "manager_loss": total_mgr_loss / self.num_regions,
            "mean_allocated_budget": float(np.mean(list(self.worker_budgets.values()))),
        }
