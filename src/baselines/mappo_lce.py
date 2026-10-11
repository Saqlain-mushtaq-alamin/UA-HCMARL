"""
MAPPO-LCE Baseline (B5) (Phase 2 §2.3)

MAPPO with PID-Lagrangian Cost Estimator (LCE).
Features:
- Extends MAPPO with 3 safety constraint penalties (GreenTime, GreenSkip, PhaseSkip).
- Dynamically updates Lagrangian multipliers via PID controller.
- Modifies advantages / rewards using constraint costs to guide policy optimization under constraints.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.baselines.mappo import MAPPOActor, MAPPOCentralizedCritic
from src.models.lagrangian import PIDLagrangianMultiplier


class MAPPOLCEController:
    """
    MAPPO-LCE Controller managing actor, reward critic, cost critic, and PID-Lagrangian multipliers.
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        share_actor: bool = True,
        num_constraints: int = 3,
        constraint_limits: Optional[List[float]] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        kp: float = 0.05,
        ki: float = 0.005,
        kd: float = 0.01,
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

        self.num_constraints = num_constraints
        # FIX: Raise constraint limits so early training isn't crushed by penalties.
        # 0.3 = allow 30% violation rate before Lagrangian activates.  Once the
        # policy learns basic traffic management, tighten these to 0.1 in later phases.
        self.constraint_limits = (
            np.array(constraint_limits, dtype=np.float32)
            if constraint_limits is not None
            else np.array([0.3, 0.3, 0.3], dtype=np.float32)
        )

        # PID-Lagrangian multiplier manager
        self.pid_lagrangian = PIDLagrangianMultiplier(
            num_constraints=num_constraints, kp=kp, ki=ki, kd=kd
        )

        # Reward Critic & Cost Critic
        self.reward_critic = MAPPOCentralizedCritic(self.global_state_dim).to(self.device)
        self.cost_critic = MAPPOCentralizedCritic(self.global_state_dim).to(self.device)
        
        self.reward_critic_optimizer = optim.Adam(self.reward_critic.parameters(), lr=lr)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=lr)

        # Actor(s)
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

    def reset(self, agent_ids=None):
        """Reset Lagrangian state between episodes to prevent multiplier carry-over."""
        self.pid_lagrangian.reset()
        self.reset_buffers()

    def reset_buffers(self):
        self.buffer = {
            "obs": {aid: [] for aid in self.agent_ids},
            "global_states": [],
            "actions": {aid: [] for aid in self.agent_ids},
            "log_probs": {aid: [] for aid in self.agent_ids},
            "rewards": [],
            "costs": [],  # per-step constraint costs
            "dones": [],
            "reward_values": [],
            "cost_values": [],
        }

    def _get_global_state(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
        vecs = [obs_dict[aid] for aid in self.agent_ids if aid in obs_dict]
        if len(vecs) < self.num_agents:
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
            r_val = float(self.reward_critic(gs_t).item())
            c_val = float(self.cost_critic(gs_t).item())

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
                self.buffer["reward_values"].append(r_val)
                self.buffer["cost_values"].append(c_val)

        return actions

    def save_model(self, path: str):
        import os
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "share_actor": self.share_actor,
            "shared_actor": self.shared_actor.state_dict() if self.share_actor else None,
            "actors": {aid: a.state_dict() for aid, a in self.actors.items()} if not self.share_actor else None,
            "reward_critic": self.reward_critic.state_dict(),
            "cost_critic": self.cost_critic.state_dict(),
            "pid_lambdas": self.pid_lagrangian.lambdas if hasattr(self, "pid_lagrangian") else None,
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
        if "reward_critic" in checkpoint:
            self.reward_critic.load_state_dict(checkpoint["reward_critic"])
        if "cost_critic" in checkpoint:
            self.cost_critic.load_state_dict(checkpoint["cost_critic"])
        if "pid_lambdas" in checkpoint and checkpoint["pid_lambdas"] is not None and hasattr(self, "pid_lagrangian"):
            self.pid_lagrangian.lambdas = checkpoint["pid_lambdas"]

    def store_transition(
        self,
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        step_violations: Optional[Dict[str, int]] = None,
    ):
        team_reward = float(np.mean([rewards[aid] for aid in self.agent_ids if aid in rewards]))
        is_done = any(dones.values())

        c_green = float(step_violations.get("green_time", 0)) if step_violations else 0.0
        c_gskip = float(step_violations.get("green_skip", 0)) if step_violations else 0.0
        c_pskip = float(step_violations.get("phase_skip", 0)) if step_violations else 0.0

        step_costs = np.array([c_green, c_gskip, c_pskip], dtype=np.float32)

        self.buffer["rewards"].append(team_reward)
        self.buffer["costs"].append(step_costs)
        self.buffer["dones"].append(is_done)

    def update(self, num_epochs: int = 4, batch_size: int = 64) -> Dict[str, float]:
        if len(self.buffer["global_states"]) == 0:
            return {"policy_loss": 0.0, "reward_value_loss": 0.0, "cost_value_loss": 0.0}

        global_states = torch.tensor(
            np.array(self.buffer["global_states"]), dtype=torch.float32, device=self.device
        )
        rewards = np.array(self.buffer["rewards"])
        costs = np.array(self.buffer["costs"])  # shape: (N, 3)
        dones = np.array(self.buffer["dones"])

        r_values = np.array(self.buffer["reward_values"])
        c_values = np.array(self.buffer["cost_values"])
        num_steps = len(rewards)

        # Aggregate step costs and update PID-Lagrangian multipliers
        # Costs are counts per step (violations), which can be 0–N per step.
        # Normalize by num_agents and num_constraints to get a rate-like quantity.
        normalization = max(1.0, float(self.num_agents * self.num_constraints))
        mean_costs_norm = (np.mean(costs, axis=0) / normalization) if num_steps > 0 else np.zeros(self.num_constraints)
        lambdas = self.pid_lagrangian.update(
            constraint_costs=mean_costs_norm,
            constraint_limits=self.constraint_limits / normalization,
        )

        total_lambda = float(np.sum(lambdas))
        total_costs_raw = np.sum(costs, axis=-1)  # scalar raw cost per step

        # FIX: Hard cap penalty at 20% of mean absolute reward magnitude per step.
        # This ensures the Lagrangian term is a guidance signal, not a dominant loss.
        mean_abs_reward = float(np.mean(np.abs(rewards))) + 1e-6
        mean_raw_cost = float(np.mean(np.abs(total_costs_raw))) + 1e-6
        # penalty_scale: how much to scale total_lambda * cost to fit target budget
        target_penalty_budget = 0.20 * mean_abs_reward  # 20% of reward magnitude
        penalty_scale = min(1.0, target_penalty_budget / (total_lambda * mean_raw_cost + 1e-6))

        # Penalized combined reward (bounded)
        penalized_rewards = rewards - penalty_scale * total_lambda * total_costs_raw

        # Compute GAE on Penalized Rewards
        advantages = np.zeros_like(penalized_rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(num_steps)):
            next_val = r_values[t + 1] if t + 1 < num_steps else 0.0
            next_non_terminal = 1.0 - float(dones[t])
            delta = penalized_rewards[t] + self.gamma * next_val * next_non_terminal - r_values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae

        returns = advantages + r_values
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        if adv_t.numel() > 1 and float(adv_t.std().item()) > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        else:
            adv_t = adv_t - adv_t.mean()

        cost_returns = torch.tensor(total_costs_raw, dtype=torch.float32, device=self.device)

        # 1. Update Critics
        self.reward_critic.train()
        self.cost_critic.train()
        for _ in range(num_epochs):
            r_pred = self.reward_critic(global_states)
            c_pred = self.cost_critic(global_states)

            r_val_loss = nn.functional.mse_loss(r_pred, ret_t)
            c_val_loss = nn.functional.mse_loss(c_pred, cost_returns)

            self.reward_critic_optimizer.zero_grad()
            r_val_loss.backward()
            nn.utils.clip_grad_norm_(self.reward_critic.parameters(), max_norm=0.5)
            self.reward_critic_optimizer.step()

            self.cost_critic_optimizer.zero_grad()
            c_val_loss.backward()
            nn.utils.clip_grad_norm_(self.cost_critic.parameters(), max_norm=0.5)
            self.cost_critic_optimizer.step()

        # 2. Update Actor(s)
        total_p_loss = 0.0
        total_ent_loss = 0.0
        update_count = 0

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

            actor = self.shared_actor if self.share_actor else self.actors[aid]
            opt = self.actor_optimizer if self.share_actor else self.actor_optimizers[aid]
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
            "reward_value_loss": r_val_loss.item(),
            "cost_value_loss": c_val_loss.item(),
            "lambda_green": float(lambdas[0]),
            "lambda_gskip": float(lambdas[1]),
            "lambda_pskip": float(lambdas[2]),
        }
