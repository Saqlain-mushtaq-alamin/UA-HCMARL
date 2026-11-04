import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.actuated import ActuatedController
from src.training.trainer import Trainer

env = SUMOEnv(max_steps=500, seed=42)
agent_ids = [f"intersection_{i}" for i in range(16)]
trainer = Trainer(env)

# Current Actuated
ctrl_curr = ActuatedController(queue_threshold=1.0)
ctrl_curr.reset(agent_ids)
res_curr = trainer.run_evaluation_episode(ctrl_curr, max_steps=500, seed=42)
print(f"Current Actuated (normalized obs compared with 1.0): queue={res_curr['avg_queue_length']:.2f}")

# Fixed Actuated: un-normalize obs[:4] * 20.0, test thresholds
for thresh in [1.0, 2.0, 3.0, 5.0]:
    class FixedActuated(ActuatedController):
        def select_action(self, agent_id: str, obs: np.ndarray) -> int:
            cur_phase = self.current_phase.get(agent_id, 0)
            t_phase = self.time_in_phase.get(agent_id, 0)
            # Un-normalize back to vehicle counts
            queues = obs[:4] * 20.0 if len(obs) >= 4 else np.zeros(4)
            ns_queue = queues[0] + queues[1]
            ew_queue = queues[2] + queues[3]

            next_phase = cur_phase
            if cur_phase == 0:  # N-S Green
                if t_phase >= self.min_green:
                    if ns_queue < self.queue_threshold or t_phase >= self.max_green or ew_queue > ns_queue + 2.0:
                        next_phase = 1
            elif cur_phase == 1:  # N-S Yellow
                if t_phase >= self.yellow_duration:
                    next_phase = 2
            elif cur_phase == 2:  # E-W Green
                if t_phase >= self.min_green:
                    if ew_queue < self.queue_threshold or t_phase >= self.max_green or ns_queue > ew_queue + 2.0:
                        next_phase = 3
            elif cur_phase == 3:  # E-W Yellow
                if t_phase >= self.yellow_duration:
                    next_phase = 0

            if next_phase != cur_phase:
                self.current_phase[agent_id] = next_phase
                self.time_in_phase[agent_id] = 0
            else:
                self.time_in_phase[agent_id] = t_phase + 1
            return next_phase

    ctrl_fix = FixedActuated(queue_threshold=thresh)
    ctrl_fix.reset(agent_ids)
    res_fix = trainer.run_evaluation_episode(ctrl_fix, max_steps=500, seed=42)
    print(f"Fixed Actuated (threshold={thresh} veh): queue={res_fix['avg_queue_length']:.2f}, switches={ctrl_fix.min_green}")
