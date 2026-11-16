import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.models.belief_encoder import GRUKalmanNet
from src.baselines.bse_wrapper import BSEControllerWrapper
from experiment_config import get_config

def main():
    print("=" * 80)
    print("DIAGNOSTIC 2: PER-FEATURE ERROR IN CLOSED LOOP")
    print("=" * 80)

    cfg = get_config("canonical")
    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]

    bse_ckpt_path = "experiments/phase3_canonical/models/bse.pt"
    bse_model = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
    data = torch.load(bse_ckpt_path, map_location="cpu", weights_only=False)
    bse_model.load_state_dict(data["bse_state"] if "bse_state" in data else data)
    bse_model.eval()

    ippo_base = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
    ippo_base.load_model("experiments/phase2_canonical_v2/models/ippo.pt")

    feature_names = [
        "Queue_N (0)", "Queue_S (1)", "Queue_E (2)", "Queue_W (3)",
        "ArrRate_0 (4)", "ArrRate_1 (5)", "ArrRate_2 (6)", "ArrRate_3 (7)",
        "Phase_0 (8)", "Phase_1 (9)", "Phase_2 (10)", "Phase_3 (11)",
        "TimeInPhase (12)", "TotalQueue (13)", "PhasePressure (14)", "Throughput (15)"
    ]

    scenarios = ["clean", "moderate", "severe", "extreme"]
    for sc in scenarios:
        env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=360, seed=100)
        wrapper = BSEControllerWrapper(base_controller=ippo_base, bse_model=bse_model, history_len=5)
        wrapper.reset(agent_ids)

        obs_dict, reset_info = env.reset(seed=100)
        action_masks = reset_info.get("action_masks", None)

        raw_errors = []
        bse_errors = []

        for step in range(360):
            # Capture clean ground truth before step
            s_true = {aid: env._get_clean_observation(aid).copy() for aid in agent_ids}
            o_raw = {aid: obs_dict[aid].copy() for aid in agent_ids}

            # Run wrapper select_actions
            # To capture b_hat, we can inspect wrapper.prev_beliefs after step
            actions = wrapper.select_actions(obs_dict, evaluate=True, step_count=step, action_masks=action_masks)
            
            for aid in agent_ids:
                b_hat = wrapper.prev_beliefs[aid].cpu().numpy()
                raw_errors.append(np.abs(o_raw[aid] - s_true[aid]))
                bse_errors.append(np.abs(b_hat - s_true[aid]))

            obs_dict, _, dones, _, info = env.step(actions)
            action_masks = info.get("action_masks", None)
            if all(dones.values()):
                break

        raw_mae = np.mean(raw_errors, axis=0)
        bse_mae = np.mean(bse_errors, axis=0)

        print(f"\nScenario: {sc.upper()}")
        print(f"{'Feature':<18s} | {'Raw MAE':<10s} | {'BSE MAE':<10s} | {'Delta (BSE - Raw)':<18s} | {'Status'}")
        print("-" * 75)
        for i, name in enumerate(feature_names):
            diff = bse_mae[i] - raw_mae[i]
            status = "WORSE (BSE distorts)" if diff > 0.005 else ("BETTER" if diff < -0.005 else "SIMILAR")
            print(f"{name:<18s} | {raw_mae[i]:10.4f} | {bse_mae[i]:10.4f} | {diff:+18.4f} | {status}")
        
        # Summarize by feature group
        q_raw, q_bse = np.mean(raw_mae[0:4]), np.mean(bse_mae[0:4])
        ph_raw, ph_bse = np.mean(raw_mae[8:12]), np.mean(bse_mae[8:12])
        tip_raw, tip_bse = raw_mae[12], bse_mae[12]
        press_raw, press_bse = raw_mae[14], bse_mae[14]

        print("-" * 75)
        print(f"Summary for {sc.upper()}:")
        print(f"  Queues (0-3)     : Raw MAE = {q_raw:.4f} -> BSE MAE = {q_bse:.4f} (diff: {q_bse - q_raw:+.4f})")
        print(f"  Phase One-Hot    : Raw MAE = {ph_raw:.4f} -> BSE MAE = {ph_bse:.4f} (diff: {ph_bse - ph_raw:+.4f})")
        print(f"  Time in Phase    : Raw MAE = {tip_raw:.4f} -> BSE MAE = {tip_bse:.4f} (diff: {tip_bse - tip_raw:+.4f})")
        print(f"  Phase Pressure   : Raw MAE = {press_raw:.4f} -> BSE MAE = {press_bse:.4f} (diff: {press_bse - press_raw:+.4f})")

if __name__ == "__main__":
    main()

