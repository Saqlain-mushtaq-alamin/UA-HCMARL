# Phase 3 Remediation Report: B2 — Kalman-Style Belief Recurrence in BSE Wrapper

## 1. Original Defect Summary
In `reports/phase3_independent_blockers.md` (Defect B2), the independent audit established that:
- `BSEControllerWrapper.select_actions()` in `src/baselines/bse_wrapper.py` passed `prev_belief=None` into `self.bse(hist_batch_t, prev_belief=None)`.
- Although `GRUKalmanNet.forward()` supported recursive belief tracking:
  $$\hat{\mathbf{b}}_t = (1 - \mathbf{K}_t) \odot f(\hat{\mathbf{b}}_{t-1}) + \mathbf{K}_t \odot \mathbf{o}_t$$
  the wrapper initialized a default zero vector on every step, discarding the recursive belief $\hat{\mathbf{b}}_{t-1}$.
- Consequently, the state estimator was functioning as a memoryless sliding-window feedforward neural filter rather than a recursive Bayesian filter.

---

## 2. Exact Source Files & Functions Changed
1. **`src/baselines/bse_wrapper.py`**:
   - `BSEControllerWrapper.__init__()`: Added `self.prev_beliefs: dict[str, torch.Tensor] = {}` to maintain agent-level belief states across steps.
   - `BSEControllerWrapper.reset()`: Added clean re-initialization of `self.prev_beliefs.clear()` on episode boundaries, ensuring no cross-episode belief leakage.
   - `BSEControllerWrapper.select_actions()`:
     - Assembles batched previous belief tensor:
       ```python
       prev_belief_batch = torch.stack([
           self.prev_beliefs[aid] if aid in self.prev_beliefs
           else torch.zeros(self.bse.belief_dim, dtype=torch.float32)
           for aid in active_ids
       ])
       ```
     - Evaluates recursive filtering:
       ```python
       b_hat_t, cov_t, k_gain_t = self.bse(hist_batch_t, prev_belief=prev_belief_batch)
       ```
     - Updates agent belief store with detached tensors:
       ```python
       for idx, aid in enumerate(active_ids):
           self.prev_beliefs[aid] = b_hat_t[idx].detach().clone()
       ```
     - Strictly preserves batched GPU/CPU tensor forward pass ($16 \times 5 \times 16$), `torch.no_grad()`, `eval()` mode, and action masking pass-through.

---

## 3. Implementation Details & Recurrence Equation
The GRUKalmanNet implementation implements the discrete-time continuous-state nonlinear filtering equation:
1. Prior State Propagation:
   $$\mathbf{h}_t = \text{GRUCell}(\mathbf{o}_t, \mathbf{h}_{t-1})$$
   $$\mathbf{b}_{t|t-1} = \mathbf{W}_p \mathbf{h}_t + \mathbf{b}_p$$
2. Kalman Gain Synthesis:
   $$\mathbf{K}_t = \sigma(\mathbf{W}_K [\mathbf{h}_t, \mathbf{b}_{t|t-1}, \hat{\mathbf{b}}_{t-1}])$$
3. Posterior State Correction:
   $$\hat{\mathbf{b}}_t = (1 - \mathbf{K}_t) \odot \mathbf{b}_{t|t-1} + \mathbf{K}_t \odot \mathbf{o}_t$$
4. State Evolution Retention:
   By storing $\hat{\mathbf{b}}_t$ in `self.prev_beliefs[agent_id]` and feeding it into $t+1$, belief history accumulates recursively over the entire episode horizon ($H=3,600$ steps), matching the Kalman filter formulation.

---

## 4. Tests Added & Verification
- **Test File**: `tests/test_bse_wrapper.py`
  - `test_kalman_recurrence_affects_updated_belief`: Confirmed that setting `prev_belief` alters output belief $\hat{\mathbf{b}}_t$ by $> 0.10$ $L_2$ norm relative to zero prior, proving active recurrence.
  - `test_prior_belief_passed_and_updated_each_step`: Verified that `self.prev_beliefs[aid]` updates at each step and retains state across sequential calls.
  - `test_intersection_isolation_no_cross_agent_leakage`: Proved belief tensors are maintained strictly per-intersection without cross-agent contamination.
  - `test_no_cross_episode_state_survives_reset`: Proved `wrapper.reset()` purges all stored beliefs.

### Verification Command & Output
```bash
python -m pytest tests/test_bse_wrapper.py -v
```
Output:
```
tests/test_bse_wrapper.py::test_bse_wrapper_initialization PASSED
tests/test_bse_wrapper.py::test_bse_wrapper_action_selection_shape PASSED
tests/test_bse_wrapper.py::test_bse_wrapper_batched_evaluation PASSED
tests/test_bse_wrapper.py::test_bse_wrapper_history_padding PASSED
tests/test_bse_wrapper.py::test_bse_wrapper_reset PASSED
tests/test_bse_wrapper.py::test_kalman_recurrence_affects_updated_belief PASSED
tests/test_bse_wrapper.py::test_prior_belief_passed_and_updated_each_step PASSED
tests/test_bse_wrapper.py::test_intersection_isolation_no_cross_agent_leakage PASSED
tests/test_bse_wrapper.py::test_no_cross_episode_state_survives_reset PASSED
============================== 9 passed in 0.98s ===============================
```

---

## 5. Remaining Limitations
1. In the first step of an episode ($t=0$), $\hat{\mathbf{b}}_{-1}$ is initialized to the zero vector before transitioning to recursive updates.
2. In extreme sensor blackout conditions, belief relies solely on prior prediction $\mathbf{b}_{t|t-1}$ from GRU dynamics.

---

## 6. Evidence Paths
- `src/baselines/bse_wrapper.py` (lines 35–45, 65–75, 115–140)
- `tests/test_bse_wrapper.py` (lines 140–235)

