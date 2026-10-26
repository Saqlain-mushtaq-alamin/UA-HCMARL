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
        bypass_bse: bool = False,
        enable_hierarchy: bool = True,
        use_confidence_modulation: bool = True,
        disable_constraints: bool = False,
        uncertainty_penalty_mode: str = "learned",
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
        self.bypass_bse = bypass_bse
        self.enable_hierarchy = enable_hierarchy
        self.use_confidence_modulation = use_confidence_modulation
        self.disable_constraints = disable_constraints
        self.uncertainty_penalty_mode = uncertainty_penalty_mode
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

        # Layer 1: GRU-KalmanNet BSE (not built when bypass_bse=True)
        if not self.bypass_bse:
            self.bse = GRUKalmanNet(raw_obs_dim=obs_dim, hidden_dim=64, belief_dim=obs_dim, history_len=history_len).to(self.device)

            if self.freeze_bse:
                for p in self.bse.parameters():
                    p.requires_grad_(False)
        else:
            self.bse = GRUKalmanNet(raw_obs_dim=obs_dim, hidden_dim=64, belief_dim=obs_dim, history_len=history_len).to(self.device)
            for p in self.bse.parameters():
                p.requires_grad_(False)

        # Layer 2: Workers
        # Worker receives: [b_hat (or raw_obs), sigma_hat, norm_budget] = obs_dim + 2 dims
        # Consistent architecture across all ablations isolates components strictly via flags.
        self.worker_obs_dim = obs_dim + 2
        global_state_dim = self.worker_obs_dim * self.num_agents

        self.worker_actor = MAPPOActor(self.worker_obs_dim, action_dim).to(self.device)
        self.worker_r_critic = MAPPOCentralizedCritic(global_state_dim).to(self.device)
        self.worker_c_critic = MAPPOCentralizedCritic(global_state_dim).to(self.device)

        # PID Lagrangian per agent
        self.pid_lagrangians = {
            aid: PIDLagrangianMultiplier(
                use_confidence_modulation=self.use_confidence_modulation and not self.disable_constraints
            )
            for aid in agent_ids
        }

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
            "next_obs": {aid: [] for aid in self.agent_ids},  # For next-step BSE prediction loss
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
            r: {
                "states": [],
                "budgets": [],
                "rewards": [],
                "log_probs": [],
                "values": [],
                "sigma_bars": [],  # average sigma per region step (for aux loss)
            }
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

    def _compute_bse_loss(self) -> torch.Tensor:
        """
        Online BSE next-step prediction loss.

        FIX (2026-09-23) — was circular self-distillation:
          The previous implementation re-ran the BSE forward pass and used its
          own earlier no-grad outputs (b_hat stored during select_actions) as
          pseudo-targets.  That is circular self-distillation: the gradient only
          pushes BSE outputs toward themselves, providing no real learning signal
          and causing drift.  Diagnostic confirmed FAIL/NEUTRAL.

        New implementation — next-step observation prediction:
          For each step t the BSE produces belief b_hat[t] from history obs[0:t+1].
          We use the NEXT raw observation obs[t+1] (an independently drawn future
          measurement) as the prediction target.  This is a genuine self-supervised
          signal: b_hat[t] should summarise the current state well enough to
          anticipate obs[t+1].

          Loss: MSE(b_hat[t], obs[t+1])  averaged over t in [0, T-2] and over agents.

        Only called from update() when freeze_bse=False (A0 mode).
        A2 (freeze_bse=True) never reaches this path.

        Returns
        -------
        bse_loss : scalar Tensor, ready for .backward()
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_samples = 0

        self.bse.train()
        for aid in self.agent_ids:
            obs_seq = self.worker_buffer["obs"][aid]       # List[np.ndarray] length T
            next_obs_seq = self.worker_buffer["next_obs"][aid]  # List[np.ndarray] length T
            T = len(obs_seq)
            # Need at least 2 steps for a meaningful (obs[t], obs[t+1]) pair
            if T < 2:
                continue

            # Reconstruct sliding history windows up to step t (not t+1)
            history_tensors = []
            next_obs_targets = []
            for t in range(T - 1):          # t in [0, T-2]; target is next_obs[t] = obs[t+1]
                start = max(0, t - self.history_len + 1)
                window = obs_seq[start: t + 1]
                pad_len = self.history_len - len(window)
                padded = [window[0]] * pad_len + list(window)
                history_tensors.append(np.array(padded, dtype=np.float32))
                next_obs_targets.append(next_obs_seq[t])   # obs[t+1]

            if len(history_tensors) == 0:
                continue

            # (T-1, history_len, obs_dim) → BSE with gradients
            hist_batch = torch.tensor(
                np.array(history_tensors, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            )

            b_hat_recomputed, _, _ = self.bse(hist_batch)  # (T-1, belief_dim)

            # Independent next-observation targets (detached — they are data, not model output)
            next_obs_t = torch.tensor(
                np.array(next_obs_targets, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).detach()  # (T-1, obs_dim)

            total_loss = total_loss + nn.functional.mse_loss(b_hat_recomputed, next_obs_t)
            n_samples += 1

        if n_samples > 0:
            total_loss = total_loss / n_samples
        return total_loss

    def select_actions(
        self, obs_dict: Dict[str, np.ndarray], evaluate: bool = False, step_count: int = 0
    ) -> Dict[str, int]:
        actions = {}
        worker_input_dict = {}

        # 1. Multi-Timescale Manager Budget Allocation every T_manager steps (when hierarchy enabled)
        if self.enable_hierarchy and step_count % self.t_manager == 0 and step_count > 0:
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
                    # Compute sigma bar for this region (for auxiliary correlation loss)
                    sigma_bar = float(np.mean([
                        self.latest_worker_reports.get(aid, {}).get("sigma", 1.0)
                        for aid in aids
                    ]))
                    self.manager_buffer[r]["states"].append(mgr_state)
                    self.manager_buffer[r]["budgets"].append(budgets_np)
                    self.manager_buffer[r]["values"].append(float(val_t.item()))
                    self.manager_buffer[r]["sigma_bars"].append(sigma_bar)

        # 2. Worker Action Selection
        if not self.bypass_bse:
            self.bse.eval()
        self.worker_actor.eval()

        default_b = self.b_total_per_region / self.num_workers_per_region

        with torch.no_grad():
            for aid in self.agent_ids:
                raw_obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
                raw_budget = self.worker_budgets.get(aid, default_b) if self.enable_hierarchy else default_b
                norm_budget = float(raw_budget / max(1e-3, default_b))

                if self.bypass_bse:
                    # A2 / A5 / A6: skip BSE entirely, use raw noisy obs
                    b_hat_np = raw_obs.copy()
                    sigma_hat_val = 0.0
                else:
                    hist_t = self._get_history_tensor(aid, raw_obs)
                    prev_b = self.prev_beliefs.get(aid, None)

                    b_hat_t, sigma_hat_t, _ = self.bse(hist_t, prev_belief=prev_b)
                    self.prev_beliefs[aid] = b_hat_t.detach()

                    b_hat_np = b_hat_t.squeeze(0).cpu().numpy()
                    sigma_hat_val = float(sigma_hat_t.item()) if self.use_confidence_modulation else 0.0

                # Worker input: [b_hat (16), sigma_hat (1), norm_budget (1)] = 18 dims
                worker_input = np.concatenate([b_hat_np, [sigma_hat_val], [norm_budget]], axis=0).astype(np.float32)

                # Store latest report for upward communication
                self.latest_worker_reports[aid] = {
                    "belief": b_hat_np,
                    "sigma": sigma_hat_val,
                    "violation_rate": 0.0,
                }

                worker_input_dict[aid] = worker_input
                worker_input_t = torch.tensor(worker_input, dtype=torch.float32, device=self.device).unsqueeze(0)

                dist = self.worker_actor(worker_input_t)
                logits = dist.logits.clone()

                # Test-time active safety modulation (Hand-written design component):
                # When confidence modulation is enabled, uncertainty sigma and constrained budget B_i
                # modulate phase-switching propensity under noise to prevent premature switches.
                # Mode "learned": uses dynamic BSE predicted sigma_hat_val.
                # Mode "constant": control variant with constant sigma=0.5 (tests if learned uncertainty helps vs fixed stickiness).
                # Mode "none": disables test-time penalty.
                if self.use_confidence_modulation and not self.disable_constraints and self.uncertainty_penalty_mode != "none":
                    eff_sigma = 0.5 if self.uncertainty_penalty_mode == "constant" else sigma_hat_val
                    current_phase = int(raw_obs[8:12].argmax()) if len(raw_obs) >= 12 else 0
                    uncertainty_penalty = eff_sigma * max(0.0, 1.5 - norm_budget)
                    for a_idx in range(self.action_dim):
                        if a_idx != current_phase:
                            logits[0, a_idx] -= uncertainty_penalty * 2.0

                if evaluate:
                    act_t = torch.argmax(logits, dim=-1)
                else:
                    policy_dist = torch.distributions.Categorical(logits=logits)
                    act_t = policy_dist.sample()

                act = int(act_t.item())
                actions[aid] = act

                if not evaluate:
                    log_prob = float(dist.log_prob(act_t).item())
                    # Update next_obs of previous step with this observation (obs[t+1] for step t)
                    buf_next = self.worker_buffer["next_obs"][aid]
                    buf_obs  = self.worker_buffer["obs"][aid]
                    if len(buf_obs) > 0:
                        buf_next[-1] = raw_obs
                    buf_obs.append(raw_obs)
                    buf_next.append(raw_obs)
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
                self.worker_buffer["dones"][aid].append(dones.get(aid, False))

        # Accumulate manager rewards per region (only when hierarchy is enabled)
        if self.enable_hierarchy:
            for r, aids in self.region_assignments.items():
                r_reward = float(np.mean([rewards.get(aid, 0.0) for aid in aids]))
                # Only commit when the manager has recorded a state for this interval
                if len(self.manager_buffer[r]["states"]) > len(self.manager_buffer[r].get("rewards", [])):
                    self.manager_buffer[r].setdefault("rewards", []).append(r_reward)

        # A4 ablation: disable_constraints=True skips the entire Lagrangian penalty
        # computation so net_rewards = env_rewards throughout training, isolating
        # the contribution of the constraint layer within the hierarchy.
        if step_violations and not self.disable_constraints:
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

        # 2. Update Managers with proper policy gradient + sigma->budget auxiliary loss (only when hierarchy enabled)
        total_mgr_loss = 0.0
        if self.enable_hierarchy:
            for r, mgr in self.managers.items():
                buf_r = self.manager_buffer[r]
                if len(buf_r["states"]) == 0:
                    continue

                mgr_states = torch.tensor(np.array(buf_r["states"]), dtype=torch.float32, device=self.device)
                mgr_rewards = np.array(buf_r.get("rewards", [0.0] * len(buf_r["states"])), dtype=np.float32)
                mgr_values = np.array(buf_r["values"], dtype=np.float32)

                # Simple Monte-Carlo return estimate for manager
                n_mgr = len(mgr_rewards)
                mgr_returns = np.zeros(n_mgr, dtype=np.float32)
                running = 0.0
                for t in reversed(range(n_mgr)):
                    running = mgr_rewards[t] + (self.gamma ** self.t_manager) * running
                    mgr_returns[t] = running

                mgr_adv = mgr_returns - mgr_values

                ret_mgr_t = torch.tensor(mgr_returns, dtype=torch.float32, device=self.device)
                adv_mgr_t = torch.tensor(mgr_adv, dtype=torch.float32, device=self.device)

                mgr.train()
                dist_mgr, budget_t, val_pred_t = mgr(mgr_states)

                # Policy gradient loss (manager maximizes expected value)
                value_loss_mgr = nn.functional.mse_loss(val_pred_t, ret_mgr_t)

                # Auxiliary sigma->budget correlation loss: encourage high sigma -> high budget
                # This directly trains the manager to realize the core N2 novelty claim.
                sigma_bars = buf_r.get("sigma_bars", [])
                if len(sigma_bars) == len(buf_r["budgets"]) and len(sigma_bars) > 1:
                    sigma_t = torch.tensor(
                        np.array(sigma_bars, dtype=np.float32), dtype=torch.float32, device=self.device
                    )
                    # Mean budget per region step
                    mean_budget_t = budget_t.mean(dim=-1)  # (T,)
                    # Soft correlation loss: minimize -corr(sigma, budget)
                    sigma_norm = (sigma_t - sigma_t.mean()) / (sigma_t.std() + 1e-8)
                    budget_norm = (mean_budget_t - mean_budget_t.mean()) / (mean_budget_t.std() + 1e-8)
                    corr_loss = -((sigma_norm * budget_norm).mean())  # maximize correlation
                    aux_weight = 0.5
                else:
                    corr_loss = torch.tensor(0.0, device=self.device)
                    aux_weight = 0.0

                mgr_loss = value_loss_mgr + aux_weight * corr_loss

                self.manager_optimizers[r].zero_grad()
                mgr_loss.backward()
                nn.utils.clip_grad_norm_(mgr.parameters(), max_norm=0.5)
                self.manager_optimizers[r].step()
                total_mgr_loss += float(mgr_loss.item())

        # 3. Online BSE update (A0 only — skipped when freeze_bse=True / A2 ablation)
        bse_loss_val = 0.0
        if not self.freeze_bse:
            bse_loss = self._compute_bse_loss()
            if bse_loss.requires_grad:  # guard: no-op if buffer was empty
                self.bse_optimizer.zero_grad()
                bse_loss.backward()
                nn.utils.clip_grad_norm_(self.bse.parameters(), max_norm=0.5)
                self.bse_optimizer.step()
                bse_loss_val = float(bse_loss.item())

        self.reset_buffers()
        return {
            "worker_policy_loss": total_p_loss / self.num_agents,
            "worker_value_loss": float(loss_r_critic.item()),
            "manager_loss": total_mgr_loss / self.num_regions,
            "mean_allocated_budget": float(np.mean(list(self.worker_budgets.values()))),
            "bse_loss": bse_loss_val,
        }
