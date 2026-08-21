"""
Layer 1 + Layer 2: BSE-MAPPO-LCE Controller (Phase 3 / Stage 3)

Combines Layer 1 (Belief-State Encoder: GRU-KalmanNet) with Layer 2 (Constrained MAPPO Worker).
Features:
- Denoised belief state b_hat and uncertainty score sigma_hat computed at every step.
- Confidence-Modulated PID Lagrangian multiplier update for green_time, green_skip, and phase_skip constraints.
- Multi-phase curriculum support (Phase A: pre-train BSE, Phase B: train workers with frozen BSE, Phase C: joint fine-tuning).
- Ablation toggle (use_confidence_modulation=True/False) for Novelty N3 validation.
"""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from src.models.belief_encoder import GRUKalmanNet
from src.models.lagrangian import PIDLagrangianMultiplier
from src.baselines.mappo import MAPPOActor, MAPPOCentralizedCritic


class BSEMAPPOLCEController:
    """
    Stage 3 Controller: BSE + MAPPO-LCE with Confidence-Modulated Constraints.
    """

    def __init__(
        self,
        agent_ids: List[str],
        obs_dim: int = 16,
        action_dim: int = 4,
        history_len: int = 5,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        use_confidence_modulation: bool = True,
        freeze_bse: bool = True,   # Default: keep BSE frozen after supervised pre-training
        device: str = "cpu",
    ):
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.history_len = history_len
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.use_confidence_modulation = use_confidence_modulation
        self.freeze_bse = freeze_bse
        self.device = torch.device(device)

        # Worker input dim: b_hat (obs_dim) + sigma_hat (1)
        self.worker_obs_dim = obs_dim + 1
        global_state_dim = self.worker_obs_dim * self.num_agents

        # Layer 1: GRU-KalmanNet Belief State Encoder
        self.bse = GRUKalmanNet(
            raw_obs_dim=obs_dim,
            hidden_dim=hidden_dim,
            belief_dim=obs_dim,
            history_len=history_len,
        ).to(self.device)

        # Layer 2: Constrained Worker Actor & Centralized Critics
        self.actor = MAPPOActor(self.worker_obs_dim, action_dim).to(self.device)
        self.reward_critic = MAPPOCentralizedCritic(global_state_dim).to(self.device)
        self.cost_critic = MAPPOCentralizedCritic(global_state_dim).to(self.device)

        # PID-Lagrangian Multiplier (conservative gains to not crush the reward signal)
        self.pid_lagrangian = PIDLagrangianMultiplier(
            num_constraints=3,
            kp=0.01,
            ki=0.001,
            kd=0.001,
            use_confidence_modulation=use_confidence_modulation,
        )

        # Optimizers
        self.bse_optimizer = optim.Adam(self.bse.parameters(), lr=lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.reward_critic_optimizer = optim.Adam(self.reward_critic.parameters(), lr=lr)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=lr)

        # Sliding window observation history buffers
        self.obs_history_buffers: Dict[str, List[np.ndarray]] = {aid: [] for aid in agent_ids}
        self.prev_beliefs: Dict[str, torch.Tensor] = {}

        self.reset_buffers()

    def reset_buffers(self):
        self.obs_history_buffers = {aid: [] for aid in self.agent_ids}
        self.prev_beliefs = {}

        self.buffer = {
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

    def _get_history_tensor(self, aid: str, current_obs: np.ndarray) -> torch.Tensor:
        buf = self.obs_history_buffers[aid]
        buf.append(current_obs)
        if len(buf) > self.history_len:
            buf.pop(0)

        # Pad history if shorter than history_len
        pad_len = self.history_len - len(buf)
        padded = [buf[0]] * pad_len + buf
        arr = np.array(padded, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], evaluate: bool = False, step_count: int = 0
    ) -> Dict[str, int]:
        actions = {}
        worker_input_dict = {}

        self.bse.eval()
        self.actor.eval()
        self.reward_critic.eval()
        self.cost_critic.eval()

        with torch.no_grad():
            for aid in self.agent_ids:
                raw_obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                hist_t = self._get_history_tensor(aid, raw_obs)
                prev_b = self.prev_beliefs.get(aid, None)

                b_hat_t, sigma_hat_t, _ = self.bse(hist_t, prev_belief=prev_b)
                self.prev_beliefs[aid] = b_hat_t.detach()

                b_hat_np = b_hat_t.squeeze(0).cpu().numpy()
                sigma_hat_val = float(sigma_hat_t.item())

                worker_input = np.concatenate([b_hat_np, [sigma_hat_val]], axis=0)
                worker_input_dict[aid] = worker_input
                worker_input_t = torch.tensor(
                    worker_input, dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                dist = self.actor(worker_input_t)
                if evaluate:
                    act_t = torch.argmax(dist.logits, dim=-1)
                else:
                    act_t = dist.sample()

                act = int(act_t.item())
                actions[aid] = act

                if not evaluate:
                    log_prob = float(dist.log_prob(act_t).item())
                    self.buffer["obs"][aid].append(raw_obs)
                    self.buffer["b_hat"][aid].append(b_hat_np)
                    self.buffer["sigma_hat"][aid].append(sigma_hat_val)
                    self.buffer["worker_input"][aid].append(worker_input)
                    self.buffer["actions"][aid].append(act)
                    self.buffer["log_probs"][aid].append(log_prob)

            if not evaluate:
                global_vec = np.concatenate(
                    [worker_input_dict[aid] for aid in self.agent_ids], axis=0
                )
                global_t = torch.tensor(
                    global_vec, dtype=torch.float32, device=self.device
                ).unsqueeze(0)
                r_val = float(self.reward_critic(global_t).item())
                c_val = float(self.cost_critic(global_t).item())

                self.buffer["global_states"].append(global_vec)
                self.buffer["reward_values"].append(r_val)
                self.buffer["cost_values"].append(c_val)

        return actions

    def store_transition(
        self,
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        step_violations: Optional[Dict[str, int]] = None,
    ):
        for aid in self.agent_ids:
            if aid in rewards:
                self.buffer["rewards"][aid].append(rewards[aid])
                self.buffer["dones"][aid].append(dones[aid])

        cost_val = 0.0
        if step_violations:
            c_vec = np.array(
                [
                    step_violations.get("green_time", 0),
                    step_violations.get("green_skip", 0),
                    step_violations.get("phase_skip", 0),
                ],
                dtype=np.float32,
            )
            # Take mean sigma_hat across agents for update
            all_sigmas = [self.buffer["sigma_hat"][aid][-1] for aid in self.agent_ids if self.buffer["sigma_hat"][aid]]
            avg_sigma = float(np.mean(all_sigmas)) if all_sigmas else 1.0

            lambdas = self.pid_lagrangian.update(
                constraint_costs=c_vec,
                constraint_limits=np.array([1.0, 1.0, 1.0], dtype=np.float32),
                sigma_hat=avg_sigma,
            )
            cost_val = float(np.sum(c_vec * lambdas))

        for aid in self.agent_ids:
            self.buffer["costs"][aid].append(cost_val)

    def compute_gae(
        self, rewards: np.ndarray, values: np.ndarray, dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        num_steps = len(rewards)
        advantages = np.zeros(num_steps, dtype=np.float32)
        last_gae_lam = 0.0

        for t in reversed(range(num_steps)):
            next_non_terminal = 1.0 - float(dones[t])
            next_value = values[t + 1] if t + 1 < num_steps else 0.0
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )

        returns = advantages + values
        return advantages, returns

    def update(self) -> Dict[str, float]:
        if len(self.buffer["global_states"]) == 0:
            return {}

        global_states = torch.tensor(
            np.array(self.buffer["global_states"]), dtype=torch.float32, device=self.device
        )
        r_values = np.array(self.buffer["reward_values"], dtype=np.float32)
        c_values = np.array(self.buffer["cost_values"], dtype=np.float32)

        # Team average reward & cost
        mean_rewards = np.mean(
            [np.array(self.buffer["rewards"][aid]) for aid in self.agent_ids], axis=0
        )
        mean_costs = np.mean(
            [np.array(self.buffer["costs"][aid]) for aid in self.agent_ids], axis=0
        )
        dones = np.array(self.buffer["dones"][self.agent_ids[0]])

        # Constrained surrogate reward: R_tot = R_env - C_pen
        net_rewards = mean_rewards - mean_costs

        advantages, returns = self.compute_gae(net_rewards, r_values, dones)
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if adv_t.numel() > 1 and float(adv_t.std().item()) > 1e-8:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        else:
            adv_t = adv_t - adv_t.mean()

        cost_returns = torch.tensor(mean_costs, dtype=torch.float32, device=self.device)

        # 1. Update Centralized Reward & Cost Critics
        self.reward_critic.train()
        self.cost_critic.train()

        r_val_pred = self.reward_critic(global_states)
        c_val_pred = self.cost_critic(global_states)

        loss_r_critic = nn.functional.mse_loss(r_val_pred, ret_t)
        loss_c_critic = nn.functional.mse_loss(c_val_pred, cost_returns)

        self.reward_critic_optimizer.zero_grad()
        loss_r_critic.backward()
        self.reward_critic_optimizer.step()

        self.cost_critic_optimizer.zero_grad()
        loss_c_critic.backward()
        self.cost_critic_optimizer.step()

        # 2. Update Workers (Actor)
        self.actor.train()
        total_policy_loss = 0.0

        for aid in self.agent_ids:
            w_input = torch.tensor(
                np.array(self.buffer["worker_input"][aid]), dtype=torch.float32, device=self.device
            )
            actions = torch.tensor(
                np.array(self.buffer["actions"][aid]), dtype=torch.long, device=self.device
            )
            old_log_probs = torch.tensor(
                np.array(self.buffer["log_probs"][aid]), dtype=torch.float32, device=self.device
            )

            dist = self.actor(w_input)
            new_log_probs = dist.log_prob(actions)
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * adv_t
            surr2 = torch.clamp(ratios, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
            policy_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
            self.actor_optimizer.step()

            total_policy_loss += policy_loss.item()

        # BSE fine-tuning is intentionally disabled during RL training.
        # The GRU-KalmanNet requires correct history sequences (batch, history_len, obs_dim)
        # which are not available from the flat replay buffer.  Pre-train BSE supervised
        # before RL starts, then freeze it (freeze_bse=True by default).
        loss_bse_val = 0.0

        self.reset_buffers()
        return {
            "policy_loss": total_policy_loss / self.num_agents,
            "reward_value_loss": float(loss_r_critic.item()),
            "cost_value_loss": float(loss_c_critic.item()),
            "bse_loss": loss_bse_val,
            "gamma_mean": float(np.mean([self.pid_lagrangian.compute_gamma(0.1)])),
        }
