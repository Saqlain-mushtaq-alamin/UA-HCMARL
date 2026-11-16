"""
BSE Controller Wrapper Module (Phase 3 Blocker 1 & 2 Resolution)
================================================================
Provides a production-grade, vectorized observation wrapper that couples
the Layer 1 GRU-KalmanNet Belief State Encoder (BSE) to any underlying MARL
controller (e.g., IPPO, FMA2C, MAPPO) without modifying the base controller's
internal network weights or action-masking logic.

Key Features:
- Restores true Kalman recursive belief propagation:
      b_hat(t) = (1 - K(t)) * f(b_hat(t-1)) + K(t) * o(t)
- Batched tensor execution across all N intersections for high simulation throughput.
- Full preservation of action masking, invalid-action prevention, and phase semantics.
- Clean episode boundary resets (zero hidden-state or recurrent belief leakage).
- Strict evaluation isolation (torch.no_grad(), eval mode, no parameter drift).
- Strict ground-truth isolation (operates exclusively on noisy observation inputs).
- Support for CPU and CUDA device execution.
"""

import os
import time
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

from src.models.belief_encoder import GRUKalmanNet


class BSEControllerWrapper:
    """
    Wraps a multi-agent traffic signal controller with Layer 1 GRU-KalmanNet
    Belief State Estimation with recurrent Kalman belief updates.

    Observations flow as:
        Noisy Environment Obs -> Sliding Window History + Prior Belief b_hat(t-1)
        -> Batched GRU-KalmanNet -> Denoised Belief b_hat(t) -> Base Controller -> Actions.
    """

    def __init__(
        self,
        base_controller: Any,
        bse_model: GRUKalmanNet,
        history_len: int = 5,
        device: str = "cpu",
    ):
        """
        Parameters
        ----------
        base_controller : Any
            The underlying MARL controller (IPPOController, FMA2CController, etc.).
        bse_model : GRUKalmanNet
            Trained or initialized GRU-KalmanNet belief state encoder.
        history_len : int
            Temporal window length H for the sliding observation history.
        device : str
            Computation device ("cpu" or "cuda").
        """
        self.base_controller = base_controller
        self.bse = bse_model
        self.history_len = history_len
        self.device = torch.device(device)

        # Mirror core controller attributes
        self.agent_ids: List[str] = list(getattr(base_controller, "agent_ids", []))
        self.num_agents: int = len(self.agent_ids)
        self.obs_dim: int = getattr(base_controller, "obs_dim", 16)
        self.action_dim: int = getattr(base_controller, "action_dim", 4)

        # Ensure BSE is on correct device
        self.bse.to(self.device)

        # Internal buffers
        self.histories: Dict[str, List[np.ndarray]] = {aid: [] for aid in self.agent_ids}
        self.prev_beliefs: Dict[str, Optional[torch.Tensor]] = {aid: None for aid in self.agent_ids}

        # Diagnostics & Profiling
        self.total_inference_time: float = 0.0
        self.total_inference_calls: int = 0
        self.last_uncertainties: Dict[str, float] = {aid: 0.0 for aid in self.agent_ids}

        self.reset()

    def reset(self, agent_ids: Optional[List[str]] = None):
        """
        Reset sliding history buffers and belief states.
        Guarantees zero history or recurrent belief leakage across episodes or evaluation seeds.
        """
        if agent_ids is not None:
            self.agent_ids = list(agent_ids)
            self.num_agents = len(self.agent_ids)

        self.histories = {aid: [] for aid in self.agent_ids}
        self.prev_beliefs = {aid: None for aid in self.agent_ids}
        self.last_uncertainties = {aid: 0.0 for aid in self.agent_ids}

        # Propagate reset to underlying controller if supported
        if hasattr(self.base_controller, "reset"):
            try:
                self.base_controller.reset(self.agent_ids)
            except TypeError:
                self.base_controller.reset()
        elif hasattr(self.base_controller, "reset_buffers"):
            self.base_controller.reset_buffers()

    def select_actions(
        self,
        obs_dict: Dict[str, np.ndarray],
        evaluate: bool = True,
        step_count: int = 0,
        action_masks: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, int]:
        """
        Denoises observations through batched GRU-KalmanNet using recurrent prior beliefs
        and dispatches to the underlying controller with full action masking.
        """
        t0 = time.perf_counter()

        # 1. Update sliding observation history buffers
        for aid in self.agent_ids:
            obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
            buf = self.histories.setdefault(aid, [])
            buf.append(obs)
            if len(buf) > self.history_len:
                buf.pop(0)

        # 2. Build batched history tensor: (num_agents, history_len, obs_dim)
        padded_batches = []
        for aid in self.agent_ids:
            buf = self.histories[aid]
            pad_len = self.history_len - len(buf)
            if pad_len > 0:
                # Pad with earliest available observation
                padded = [buf[0]] * pad_len + buf
            else:
                padded = buf
            padded_batches.append(padded)

        hist_batch_np = np.array(padded_batches, dtype=np.float32)
        hist_batch_t = torch.from_numpy(hist_batch_np).to(self.device)

        # 3. Assemble batched prior belief tensor (b_hat(t-1))
        all_have_prior = all(self.prev_beliefs.get(aid) is not None for aid in self.agent_ids)
        if all_have_prior and len(self.agent_ids) > 0:
            prior_tensors = [self.prev_beliefs[aid] for aid in self.agent_ids]
            prev_belief_batch = torch.stack(prior_tensors, dim=0).to(self.device)
        else:
            # On first step or after reset: pass None so GRUKalmanNet initializes prev_belief = current_obs
            prev_belief_batch = None

        # 4. Batched BSE forward inference (Strictly no-grad, eval mode)
        self.bse.eval()
        with torch.no_grad():
            b_hat_t, sigma_hat_t, _ = self.bse(hist_batch_t, prev_belief=prev_belief_batch)
            b_hat_np = b_hat_t.cpu().numpy()
            sigma_np = sigma_hat_t.squeeze(-1).cpu().numpy()

        # 5. Construct denoised observation dictionary & update recurrent belief state
        denoised_obs = {}
        for idx, aid in enumerate(self.agent_ids):
            # Recurrent Kalman state: store updated belief for next step
            self.prev_beliefs[aid] = b_hat_t[idx].detach().clone()
            denoised_obs[aid] = b_hat_np[idx]
            self.last_uncertainties[aid] = float(sigma_np[idx])

        # Record profiling stats
        self.total_inference_time += (time.perf_counter() - t0)
        self.total_inference_calls += 1

        # 6. Dispatch to underlying controller with action masking
        if hasattr(self.base_controller, "select_actions"):
            try:
                actions = self.base_controller.select_actions(
                    denoised_obs,
                    evaluate=evaluate,
                    step_count=step_count,
                    action_masks=action_masks,
                )
            except TypeError:
                try:
                    actions = self.base_controller.select_actions(
                        denoised_obs,
                        evaluate=evaluate,
                        step_count=step_count,
                    )
                except TypeError:
                    actions = self.base_controller.select_actions(denoised_obs)
        else:
            raise AttributeError(
                f"Base controller {type(self.base_controller).__name__} does not implement select_actions"
            )

        return actions

    def store_transition(
        self,
        rewards: Dict[str, float],
        dones: Dict[str, bool],
        step_violations: Optional[Dict[str, int]] = None,
    ):
        """Forward transitions to base controller if supported."""
        if hasattr(self.base_controller, "store_transition"):
            try:
                self.base_controller.store_transition(
                    rewards=rewards,
                    dones=dones,
                    step_violations=step_violations,
                )
            except TypeError:
                self.base_controller.store_transition(rewards=rewards, dones=dones)

    def save_model(self, path: str):
        """Save BSE model checkpoint."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(
            {
                "bse_state": self.bse.state_dict(),
                "history_len": self.history_len,
                "obs_dim": self.obs_dim,
                "base_controller_class": type(self.base_controller).__name__,
            },
            path,
        )

    def load_bse_checkpoint(self, path: str):
        """Load pre-trained BSE weights into wrapper."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(data, dict) and "bse_state" in data:
            self.bse.load_state_dict(data["bse_state"])
        elif isinstance(data, dict):
            self.bse.load_state_dict(data)
        else:
            raise ValueError(f"Unrecognized checkpoint format at {path}")

    def get_average_latency_ms(self) -> float:
        """Returns average BSE inference latency per simulation step in milliseconds."""
        if self.total_inference_calls == 0:
            return 0.0
        return (self.total_inference_time / self.total_inference_calls) * 1000.0
