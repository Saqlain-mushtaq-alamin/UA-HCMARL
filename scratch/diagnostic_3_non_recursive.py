import os
import sys
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

class NonRecursiveBSEWrapper(BSEControllerWrapper):
    """BSE Wrapper that drops recurrent belief propagation (prev_belief=None)."""
    def select_actions(self, obs_dict, evaluate=True, step_count=0, action_masks=None):
        # Update sliding observation history buffers
        for aid in self.agent_ids:
            obs = obs_dict.get(aid, np.zeros(self.obs_dim, dtype=np.float32))
            buf = self.histories.setdefault(aid, [])
            buf.append(obs)
            if len(buf) > self.history_len:
                buf.pop(0)

        padded_batches = []
        for aid in self.agent_ids:
            buf = self.histories[aid]
            pad_len = self.history_len - len(buf)
            if pad_len > 0:
                padded = [buf[0]] * pad_len + buf
            else:
                padded = buf
            padded_batches.append(padded)

        hist_batch_t = torch.tensor(np.array(padded_batches), dtype=torch.float32, device=self.device)

        # Drop prev_belief (set to None), exactly as in supervised training!
        self.bse.eval()
        with torch.no_grad():
            b_hat_t, sigma_hat_t, _ = self.bse(hist_batch_t, prev_belief=None)
            b_hat_np = b_hat_t.cpu().numpy()

        denoised_obs = {aid: b_hat_np[i] for i, aid in enumerate(self.agent_ids)}

        if hasattr(self.base_controller, "select_actions"):
            return self.base_controller.select_actions(
                denoised_obs, evaluate=evaluate, step_count=step_count, action_masks=action_masks
            )
        raise AttributeError("No select_actions")

def main():
    print("=" * 80)
    print("DIAGNOSTIC 3: NON-RECURSIVE BSE AT INFERENCE (DROP prev_belief)")
    print("=" * 80)

    cfg = get_config("canonical")
    agent_ids = [f"intersection_{i}" for i in range(cfg.num_intersections)]

    bse_ckpt_path = "experiments/phase3_canonical/models/bse.pt"
    bse_model = GRUKalmanNet(raw_obs_dim=16, hidden_dim=64, belief_dim=16, history_len=5)
    data = torch.load(bse_ckpt_path, map_location="cpu", weights_only=False)
    bse_model.load_state_dict(data["bse_state"] if "bse_state" in data else data)
    bse_model.eval()

    eval_seeds = [100, 101]
    scenarios = ["clean", "severe", "extreme"]

    for sc in scenarios:
        print(f"\n================ Scenario: {sc.upper()} ================")
        
        # Base controllers
        m1_clean = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
        m1_clean.load_model("experiments/phase2_canonical_v2/models/ippo.pt")

        m3_rec = BSEControllerWrapper(base_controller=m1_clean, bse_model=bse_model, history_len=5)
        
        m1_clean_2 = IPPOController(agent_ids=agent_ids, obs_dim=16, action_dim=4, share_actor=True, share_critic=True)
        m1_clean_2.load_model("experiments/phase2_canonical_v2/models/ippo.pt")
        m3_nonrec = NonRecursiveBSEWrapper(base_controller=m1_clean_2, bse_model=bse_model, history_len=5)

        m4_clean = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
        m4_clean.load_model("experiments/phase2_canonical_v2/models/fma2c.pt")

        m6_rec = BSEControllerWrapper(base_controller=m4_clean, bse_model=bse_model, history_len=5)

        m4_clean_2 = FMA2CController(agent_ids=agent_ids, obs_dim=16, action_dim=4, num_regions=min(4, cfg.num_intersections))
        m4_clean_2.load_model("experiments/phase2_canonical_v2/models/fma2c.pt")
        m6_nonrec = NonRecursiveBSEWrapper(base_controller=m4_clean_2, bse_model=bse_model, history_len=5)

        controllers = {
            "M1: IPPO Clean": m1_clean,
            "M3: IPPO + BSE (Recursive)": m3_rec,
            "M3: IPPO + BSE (Non-Rec)": m3_nonrec,
            "M4: FMA2C Clean": m4_clean,
            "M6: FMA2C + BSE (Recursive)": m6_rec,
            "M6: FMA2C + BSE (Non-Rec)": m6_nonrec,
        }

        results = {name: [] for name in controllers}
        for ep_seed in eval_seeds:
            for name, ctrl in controllers.items():
                env = SUMOEnv(num_intersections=cfg.num_intersections, noise_scenario=sc, max_steps=cfg.max_steps, seed=ep_seed)
                trainer = Trainer(env=env)
                if hasattr(ctrl, "reset"):
                    ctrl.reset(agent_ids)
                res = trainer.run_evaluation_episode(ctrl, max_steps=cfg.max_steps, seed=ep_seed)
                results[name].append(res["avg_delay"])

        for name, vals in results.items():
            print(f"  {name:30s}: Mean = {np.mean(vals):6.2f} +/- {np.std(vals):5.2f} (runs: {[round(x, 2) for x in vals]})")

if __name__ == "__main__":
    main()
