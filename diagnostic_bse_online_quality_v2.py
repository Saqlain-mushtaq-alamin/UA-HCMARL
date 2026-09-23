"""
diagnostic_bse_online_quality_v2.py  (Step 1 re-run with fixed _compute_bse_loss)

Same as the original diagnostic, but now _compute_bse_loss uses next-step
observation prediction (a real self-supervised signal) instead of the circular
self-distillation that was confirmed NEUTRAL.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.ua_hcmarl import UAHCMARLController
from src.training.trainer import Trainer
from src.training.bse_trainer import (
    collect_bse_dataset,
    train_bse_supervised,
    evaluate_bse_calibration,
)

N_AGENTS = 8
agent_ids = [f"intersection_{i}" for i in range(N_AGENTS)]

print("=" * 60)
print("  BSE Online-Update Quality Diagnostic v2 (fixed loss)")
print("=" * 60)

env = SUMOEnv(num_intersections=N_AGENTS, max_steps=500, seed=42)
trainer = Trainer(env=env)

ctrl = UAHCMARLController(agent_ids=agent_ids, freeze_bse=False)

print("\n[1/4] Collecting supervised pre-training dataset (1500 steps)...")
X_tr, Y_tr = collect_bse_dataset(env, num_steps=1500)
print(f"      Dataset: X={X_tr.shape}, Y={Y_tr.shape}")

print("[2/4] Pre-training BSE (supervised, 15 epochs)...")
train_bse_supervised(ctrl.bse, X_tr, Y_tr, epochs=15)

print("[3/4] Collecting held-out calibration set (400 steps)...")
X_te, Y_te = collect_bse_dataset(env, num_steps=400)

calib_before = evaluate_bse_calibration(ctrl.bse, X_te, Y_te)
print(f"\n  BEFORE online RL:")
print(f"    bse_mse = {calib_before['bse_mse']:.6f}")
print(f"    raw_mse = {calib_before['raw_mse']:.6f}")
print(f"    corr    = {calib_before['sigma_calibration_correlation']:.4f}")
print(f"    MSE reduction vs raw: {calib_before['mse_reduction_pct']:.1f}%")

print("\n[4/4] Running 15 RL training episodes (FIXED online BSE update: next-step pred)...")
for ep in range(15):
    summary, loss_info = trainer.train_episode(ctrl, max_steps=500, seed=42 + ep)
    bse_loss = loss_info.get("bse_loss", float("nan"))
    print(f"  ep {ep+1:2d}: avg_delay={summary['avg_delay']:.2f}  bse_loss={bse_loss:.6f}")

calib_after = evaluate_bse_calibration(ctrl.bse, X_te, Y_te)
print(f"\n  AFTER online RL:")
print(f"    bse_mse = {calib_after['bse_mse']:.6f}")
print(f"    raw_mse = {calib_after['raw_mse']:.6f}")
print(f"    corr    = {calib_after['sigma_calibration_correlation']:.4f}")
print(f"    MSE reduction vs raw: {calib_after['mse_reduction_pct']:.1f}%")

ratio = calib_after['bse_mse'] / max(calib_before['bse_mse'], 1e-9)
print(f"\n  MSE ratio (after/before): {ratio:.4f}")
print("=" * 60)

if calib_after['bse_mse'] > calib_before['bse_mse'] * 1.10:
    verdict = "FAIL"
    note = "Fixed loss still harmful — investigate further."
elif calib_after['bse_mse'] < calib_before['bse_mse'] * 0.95:
    verdict = "PASS"
    note = "Next-step prediction improved BSE vs ground truth — proceed to Phase 5."
else:
    verdict = "NEUTRAL"
    note = "Next-step prediction had negligible effect. Acceptable — not harmful."

print(f"  VERDICT: {verdict}")
print(f"  {note}")
print("=" * 60)
