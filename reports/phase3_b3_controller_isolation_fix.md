# Phase 3 Remediation Report: B3 — Controller Instance Isolation & Order Invariance

## 1. Original Defect Summary
In `reports/phase3_independent_blockers.md` (Defect B3), the independent audit established that:
- In `scripts/run_phase3_benchmark.py`, Arm M3 (`IPPO + BSE`) wrapped the exact same controller instance as Arm M1 (`Clean IPPO`), and Arm M6 (`FMA2C + BSE`) wrapped the exact same controller instance as Arm M4 (`Clean FMA2C`).
- In FMA2C, regional consensus goals (`self.current_goals`) and hidden consensus buffers are stateful across rollout steps.
- Evaluating Arm M4 followed by Arm M6 mutated the underlying controller state, causing evaluation results to be dependent on execution order and risking state contamination between benchmark arms.

---

## 2. Exact Source Files & Functions Changed
1. **`src/baselines/fma2c.py`**:
   - Added standard `reset(self, agent_ids=None)` method to `FMA2CController`:
     ```python
     def reset(self, agent_ids=None):
         """Reset stateful consensus buffers and regional goals on episode boundary."""
         self.reset_buffers()
         for aid in self.agent_ids:
             self.current_goals[aid] = np.zeros(self.action_dim, dtype=np.float32)
     ```
2. **`scripts/run_phase3_benchmark.py`**:
   - Helper function `get_clean_controller(alg, seed)` instantiated to load fresh, isolated instances from canonical Phase 2 checkpoints.
   - Arms M1, M3, M4, and M6 each call `get_clean_controller()` separately:
     ```python
     # Arm M1: Clean IPPO
     m1_ippo = get_clean_controller("ippo", seed=train_seed)

     # Arm M3: Clean IPPO + BSE Wrapper (Distinct controller instance from M1)
     m3_ippo_base = get_clean_controller("ippo", seed=train_seed)
     m3_bse = BSEControllerWrapper(base_controller=m3_ippo_base, bse_model=bse_model, history_len=5)

     # Arm M4: Clean FMA2C
     m4_fma2c = get_clean_controller("fma2c", seed=train_seed)

     # Arm M6: Clean FMA2C + BSE Wrapper (Distinct controller instance from M4)
     m6_fma2c_base = get_clean_controller("fma2c", seed=train_seed)
     m6_bse = BSEControllerWrapper(base_controller=m6_fma2c_base, bse_model=bse_model, history_len=5)
     ```
   - All evaluation loops call `controller.reset()` at episode boundaries.

---

## 3. Implementation Details & Isolation Proof
- `id(m1_ippo) != id(m3_bse.base_controller)`: Memory addresses verify non-identical Python objects.
- `id(m4_fma2c) != id(m6_bse.base_controller)`: Independent weights and consensus buffers.
- Modifying parameters or internal buffers in `m3_bse.base_controller` produces strictly zero change in `m1_ippo`.
- Reversing evaluation order (`M4` then `M6` vs `M6` then `M4`) produces bit-identical metric trajectories when isolated instances and `reset()` are utilized.

---

## 4. Tests Added & Verification
- **Test File**: `tests/test_controller_isolation.py`
  - `test_independent_instances_for_bse_arms`: Asserts `m1 is not m3.base_controller` and `m4 is not m6.base_controller`, and verifies parameter mutation in one does not alter the other.
  - `test_order_invariance_across_arms`: Simulates evaluation order swapping (A then B vs B then A) on FMA2C with mutated consensus goals, verifying strict order invariance and deterministic reproducibility.

### Verification Command & Output
```bash
python -m pytest tests/test_controller_isolation.py -v
```
Output:
```
tests/test_controller_isolation.py::test_independent_instances_for_bse_arms PASSED
tests/test_controller_isolation.py::test_order_invariance_across_arms PASSED
============================== 2 passed in 0.72s ===============================
```

---

## 5. Remaining Limitations
None. Independent instance creation ensures total isolation between experimental arms.

---

## 6. Evidence Paths
- `src/baselines/fma2c.py` (lines 151–160)
- `scripts/run_phase3_benchmark.py` (lines 335–385)
- `tests/test_controller_isolation.py` (lines 1–85)

