"""
diagnostic_bse_online_quality.py  (Step 1)

Checks whether the online BSE self-distillation loss in _compute_bse_loss()
actually improves or degrades ground-truth calibration.

Expected verdict:
  FAIL   -> bse_mse after online RL > before * 1.10
  PASS   -> bse_mse after online RL < before * 0.95
  NEUTRAL-> anything else
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
print("  BSE Online-Update Quality Diagnostic")
print("=" * 60)

env = SUMOEnv(num_intersections=N_AGENTS, max_steps=500, seed=42)
trainer = Trainer(env=env)

# Build controller with online BSE enabled (freeze_bse=False = A0 mode)
ctrl = UAHCMARLController(agent_ids=agent_ids, freeze_bse=False)

# --- Supervised pre-training ---
print("\n[1/4] Collecting supervised pre-training dataset (1500 steps)...")
X_tr, Y_tr = collect_bse_dataset(env, num_steps=1500)
print(f"      Dataset: X={X_tr.shape}, Y={Y_tr.shape}")

print("[2/4] Pre-training BSE (supervised, 15 epochs)...")
train_bse_supervised(ctrl.bse, X_tr, Y_tr, epochs=15)

# --- Ground-truth calibration BEFORE online RL ---
print("[3/4] Collecting held-out calibration set (400 steps)...")
X_te, Y_te = collect_bse_dataset(env, num_steps=400)

calib_before = evaluate_bse_calibration(ctrl.bse, X_te, Y_te)
print(f"\n  BEFORE online RL:")
print(f"    bse_mse = {calib_before['bse_mse']:.6f}")
print(f"    raw_mse = {calib_before['raw_mse']:.6f}")
print(f"    corr    = {calib_before['sigma_calibration_correlation']:.4f}")
print(f"    MSE reduction vs raw: {calib_before['mse_reduction_pct']:.1f}%")

# --- Online RL training (where _compute_bse_loss fires) ---
print("\n[4/4] Running 15 RL training episodes (online BSE update active)...")
for ep in range(15):
    summary, loss_info = trainer.train_episode(ctrl, max_steps=500, seed=42 + ep)
    bse_loss = loss_info.get("bse_loss", float("nan"))
    print(f"  ep {ep+1:2d}: avg_delay={summary['avg_delay']:.2f}  bse_loss={bse_loss:.6f}")

# --- Ground-truth calibration AFTER online RL (same held-out X_te, Y_te) ---
calib_after = evaluate_bse_calibration(ctrl.bse, X_te, Y_te)
print(f"\n  AFTER online RL:")
print(f"    bse_mse = {calib_after['bse_mse']:.6f}")
print(f"    raw_mse = {calib_after['raw_mse']:.6f}")
print(f"    corr    = {calib_after['sigma_calibration_correlation']:.4f}")
print(f"    MSE reduction vs raw: {calib_after['mse_reduction_pct']:.1f}%")

# --- Verdict ---
ratio = calib_after['bse_mse'] / max(calib_before['bse_mse'], 1e-9)
print(f"\n  MSE ratio (after/before): {ratio:.4f}")
print("=" * 60)

if calib_after['bse_mse'] > calib_before['bse_mse'] * 1.10:
    verdict = "FAIL"
    note = ("Online update made BSE WORSE vs ground truth — "
            "self-distillation loss is harmful. Must fix _compute_bse_loss.")
elif calib_after['bse_mse'] < calib_before['bse_mse'] * 0.95:
    verdict = "PASS"
    note = "Online update improved BSE vs ground truth — proceed."
else:
    verdict = "NEUTRAL"
    note = ("Online update had negligible effect — "
            "drift without benefit. Fix _compute_bse_loss.")

print(f"  VERDICT: {verdict}")
print(f"  {note}")
print("=" * 60)
