"""
Diagnostic script for Step 1:
Instrument A3's sigma_hat and gamma values directly during clean, severe, and extreme noise evaluation.
Checks whether:
1) sigma_hat is elevated under severe/extreme noise vs clean
2) compute_gamma() returns 1.0 pinned regardless of sigma_hat when use_confidence_modulation=False
3) For comparison, checks A1 (use_confidence_modulation=True) to verify gamma varies as expected.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch

from src.envs.sumo_env import SUMOEnv
from src.baselines.bse_mappo_lce import BSEMAPPOLCEController
from src.training.trainer import Trainer
from src.training.bse_trainer import collect_bse_dataset, train_bse_supervised

def run_diagnostic():
    net_file = "networks/grid4x4/grid4x4.net.xml"
    route_file = "networks/grid4x4/demand_uniform.rou.xml"
    num_intersections = 16
    agent_ids = [f"intersection_{i}" for i in range(num_intersections)]
    noise_scenarios = ["clean", "mild", "moderate", "severe", "extreme"]
    max_steps = 200

    print("=" * 70)
    print("  STEP 1 DIAGNOSTIC: Instrumenting A3's sigma_hat and gamma")
    print("=" * 70)

    # 1. Initialize environment & controllers
    env_train = SUMOEnv(
        net_file=net_file,
        route_file=route_file,
        num_intersections=num_intersections,
        max_steps=max_steps,
        seed=42,
    )
    trainer = Trainer(env=env_train)

    print("\n[1/4] Initializing A3 (use_confidence_modulation=False) and A1 (use_confidence_modulation=True)...")
    a3_no_coupling = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=False)
    a1_no_hier = BSEMAPPOLCEController(agent_ids=agent_ids, use_confidence_modulation=True)

    print(f"  A3 lagrangian use_confidence_modulation: {a3_no_coupling.pid_lagrangian.use_confidence_modulation}")
    print(f"  A1 lagrangian use_confidence_modulation: {a1_no_hier.pid_lagrangian.use_confidence_modulation}")

    # 2. Supervised pre-training of BSE
    print("\n[2/4] Pre-training BSE with supervised data (300 steps, 3 epochs)...")
    X_tr, Y_tr = collect_bse_dataset(env_train, num_steps=300, noise_scenarios=noise_scenarios)
    train_bse_supervised(a3_no_coupling.bse, X_tr, Y_tr, epochs=3)
    train_bse_supervised(a1_no_hier.bse, X_tr, Y_tr, epochs=3)

    # 3. Train controllers for 3 curriculum episodes
    print("\n[3/4] Running 3 curriculum training episodes...")
    for ep in range(3):
        trainer.train_episode(a3_no_coupling, max_steps=max_steps, seed=42 + ep)
        trainer.train_episode(a1_no_hier, max_steps=max_steps, seed=42 + ep)
        print(f"  Finished train episode {ep+1}/3")

    # 4. Evaluation under Clean, Severe, and Extreme noise
    print("\n[4/4] Evaluating under Clean, Severe, and Extreme Noise with Direct Instrumentation...")

    for noise in ["clean", "severe", "extreme"]:
        print(f"\n" + "-" * 60)
        print(f"  EVALUATION SCENARIO: [{noise.upper()}]")
        print("-" * 60)

        env_eval = SUMOEnv(
            net_file=net_file,
            route_file=route_file,
            num_intersections=num_intersections,
            noise_scenario=noise,
            max_steps=max_steps,
            seed=100 if noise == "clean" else (30000 if noise == "severe" else 40000),
        )

        for ctrl_name, ctrl in [("A3: No-Coupling (Modulation=False)", a3_no_coupling),
                                ("A1: No-Hierarchy (Modulation=True)", a1_no_hier)]:
            obs_dict, _ = env_eval.reset(seed=42)
            ctrl.reset_buffers()

            sigma_hat_samples = []
            gamma_samples = []

            for step in range(max_steps):
                # Step evaluation and extract sigma_hat directly from BSE
                actions = {}
                step_sigmas = []
                step_gammas = []

                ctrl.bse.eval()
                ctrl.actor.eval()

                with torch.no_grad():
                    for aid in ctrl.agent_ids:
                        raw_obs = obs_dict.get(aid, np.zeros(ctrl.obs_dim, dtype=np.float32))
                        hist_t = ctrl._get_history_tensor(aid, raw_obs)
                        prev_b = ctrl.prev_beliefs.get(aid, None)

                        b_hat_t, sigma_hat_t, _ = ctrl.bse(hist_t, prev_belief=prev_b)
                        ctrl.prev_beliefs[aid] = b_hat_t.detach()

                        sigma_val = float(sigma_hat_t.item())
                        step_sigmas.append(sigma_val)

                        gamma_val = ctrl.pid_lagrangian.compute_gamma(sigma_val, budget=50.0)
                        step_gammas.append(gamma_val)

                        b_hat_np = b_hat_t.squeeze(0).cpu().numpy()
                        worker_input = np.concatenate([b_hat_np, [sigma_val]], axis=0)
                        worker_input_t = torch.tensor(worker_input, dtype=torch.float32).unsqueeze(0)
                        dist = ctrl.actor(worker_input_t)
                        act_t = torch.argmax(dist.logits, dim=-1)
                        actions[aid] = int(act_t.item())

                obs_dict, _, dones, _, _ = env_eval.step(actions)

                mean_step_sigma = float(np.mean(step_sigmas))
                mean_step_gamma = float(np.mean(step_gammas))
                sigma_hat_samples.append(mean_step_sigma)
                gamma_samples.append(mean_step_gamma)

                if all(dones.values()):
                    break

            sig_arr = np.array(sigma_hat_samples)
            gam_arr = np.array(gamma_samples)

            print(f"\n  Controller: {ctrl_name}")
            print(f"    {ctrl_name[:2]} sigma_hat samples (first 10 steps): {list(np.round(sig_arr[:10], 4))}")
            print(f"    {ctrl_name[:2]} gamma samples (first 10 steps):     {list(np.round(gam_arr[:10], 4))}")
            print(f"    Full episode stats ({len(sig_arr)} steps):")
            print(f"      sigma_hat : mean={sig_arr.mean():.4f}, std={sig_arr.std():.4f}, min={sig_arr.min():.4f}, max={sig_arr.max():.4f}")
            print(f"      gamma     : mean={gam_arr.mean():.4f}, std={gam_arr.std():.4f}, min={gam_arr.min():.4f}, max={gam_arr.max():.4f}")

            all_gamma_one = np.allclose(gam_arr, 1.0, atol=1e-5)
            print(f"      Is gamma pinned at exactly 1.0? {'YES (True uniform budgeting)' if all_gamma_one else 'NO (Varying with sigma)'}")

if __name__ == "__main__":
    run_diagnostic()
