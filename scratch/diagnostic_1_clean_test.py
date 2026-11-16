import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.abspath("."))
from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController
from src.baselines.fma2c import FMA2CController
from src.baselines.bse_wrapper import BSEControllerWrapper
from src.models.belief_encoder import GRUKalmanNet
from src.training.trainer import Trainer
from experiment_config import get_config

def main():
    print("=" * 80)
    print("DIAGNOSTIC 1: CLEAN-INPUT TEST & PER-SEED VALUES FOR M2 / M5 (CHECK 4)")
    print("=" * 80)

    # 1. Print per-seed values for M2 and M5 across scenarios
    summary_path = "experiments/phase3_canonical/phase3_statistical_summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        print("\n--- [CHECK 4] PER-SEED REPLICATE VALUES FOR M2 AND M5 ---")
        for sc in ["clean", "mild", "moderate", "severe", "extreme"]:
            m2 = summary["scenarios"][sc]["M2: IPPO (Noise-Trained)"]["queue_length"]
            m5 = summary["scenarios"][sc]["M5: FMA2C (Noise-Trained)"]["queue_length"]
            print(f"Scenario: {sc.upper():8s}")
            print(f"  M2 (IPPO Noise-Trained) : mean={m2['mean']:5.2f} +/- {m2['std']:4.2f} | seeds [42, 43, 44]: {[round(x, 2) for x in m2['replicate_means']]}")
            print(f"  M5 (FMA2C Noise-Trained): mean={m5['mean']:5.2f} +/- {m5['std']:4.2f} | seeds [42, 43, 44]: {[round(x, 2) for x in m5['replicate_means']]}")
    else:
        print("Canonical summary not found.")

    # 2. Run clean evaluation directly on M1 vs M3 and M4 vs M6
    print("\n--- [DIAGNOSTIC 1] RUNNING CLEAN-INPUT TEST ON TRAINED CONTROLLERS ---")
    cfg = get_config("canonical")
    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]

    bse_ckpt_path = "experiments/phase3_canonical/models/bse.pt"
    bse_model = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
    data = torch.load(bse_ckpt_path, map_location="cpu", weights_only=False)
    bse_model.load_state_dict(data["bse_state"] if "bse_state" in data else data)
    bse_model.eval()

    # Test distortion of clean observations on raw feed
    env_clean = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario="clean", max_steps=cfg.max_steps, seed=100)
    obs_dict, _ = env_clean.reset(seed=100)
    
    # Check distortion on initial step
    obs_t = torch.tensor(np.array(list(obs_dict.values())), dtype=torch.float32).unsqueeze(1) # (16, 1, 16)
    with torch.no_grad():
        b_hat, _, _ = bse_model(obs_t)
        clean_obs_np = np.array(list(obs_dict.values()))
        distortion = np.abs(b_hat.numpy() - clean_obs_np)
        print(f"\nStatic Clean Obs BSE Distortion (Single Step):")
        print(f"  Mean absolute distortion per feature: {distortion.mean(axis=0)}")
        print(f"  Overall MAE: {distortion.mean():.4f}, Max absolute distortion: {distortion.max():.4f}")

    # Evaluate closed-loop on clean scenario for seed 42 models
    print("\nEvaluating closed-loop performance on CLEAN environment (3 eval seeds x 2 eps):")
    eval_seeds = [100, 101, 102]
    
    # Load M1
    m1_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
    m1_ippo.load_model("experiments/phase2_canonical_v2/models/ippo.pt")

    # M3
    m3_ippo = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
    m3_ippo.load_model("experiments/phase2_canonical_v2/models/ippo.pt")
    m3_bse = BSEControllerWrapper(base_controller=m3_ippo, bse_model=bse_model, history_len=5)

    # Load M4
    m4_fma2c = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
    m4_fma2c.load_model("experiments/phase2_canonical_v2/models/fma2c.pt")

    # M6
    m6_fma2c = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
    m6_fma2c.load_model("experiments/phase2_canonical_v2/models/fma2c.pt")
    m6_bse = BSEControllerWrapper(base_controller=m6_fma2c, bse_model=bse_model, history_len=5)

    controllers = {
        "M1: IPPO (Clean)": m1_ippo,
        "M3: IPPO + BSE": m3_bse,
        "M4: FMA2C (Clean)": m4_fma2c,
        "M6: FMA2C + BSE": m6_bse,
    }

    results = {name: [] for name in controllers}
    for seed in eval_seeds:
        for ep in range(2):
            ep_seed = seed * 10 + ep
            for name, ctrl in controllers.items():
                env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario="clean", max_steps=cfg.max_steps, seed=ep_seed)
                trainer = Trainer(env=env)
                if hasattr(ctrl, "reset"):
                    ctrl.reset(agent_ids)
                res = trainer.run_evaluation_episode(ctrl, max_steps=cfg.max_steps, seed=ep_seed)
                results[name].append(res["avg_delay"])

    print("\n--- Clean Evaluation Results (avg queue length across runs) ---")
    for name, vals in results.items():
        print(f"  {name:22s}: Mean = {np.mean(vals):.2f} +/- {np.std(vals):.2f} (runs: {[round(x, 2) for x in vals]})")

    ippo_degrad = (np.mean(results["M3: IPPO + BSE"]) - np.mean(results["M1: IPPO (Clean)"])) / np.mean(results["M1: IPPO (Clean)"]) * 100
    fma2c_degrad = (np.mean(results["M6: FMA2C + BSE"]) - np.mean(results["M4: FMA2C (Clean)"])) / np.mean(results["M4: FMA2C (Clean)"]) * 100
    print(f"\nM3 vs M1 degradation on clean: +{ippo_degrad:.1f}%")
    print(f"M6 vs M4 degradation on clean: +{fma2c_degrad:.1f}%")

if __name__ == "__main__":
    main()

