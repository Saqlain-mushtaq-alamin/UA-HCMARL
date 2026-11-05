"""
Diagnostic script to trace IPPO red-wait dynamics step-by-step.
"""
import numpy as np
from src.envs.sumo_env import SUMOEnv
from src.baselines.ippo import IPPOController

def run_trace():
    env = SUMOEnv(num_intersections=4, max_steps=500, max_red_wait=38, seed=42)
    obs, reset_info = env.reset(seed=42)
    agent_ids = list(obs.keys())
    ctrl = IPPOController(agent_ids=agent_ids, share_actor=True, share_critic=True)
    
    # Track red wait events for intersection_0
    int_0_sim = env.simulators["intersection_0"]
    
    phase_durations = []
    current_phase = int_0_sim.current_phase
    current_dur = 0
    max_red_wait_seen = 0.0
    red_wait_history = []
    
    for step in range(300):
        action_masks = {aid: env.simulators[aid].get_action_mask() for aid in agent_ids}
        actions = ctrl.select_actions(obs, evaluate=True, step_count=step, action_masks=action_masks)
        obs, rewards, dones, truncateds, info = env.step(actions)
        
        sim = env.simulators["intersection_0"]
        rw = np.max(sim.approach_red_wait)
        if rw > max_red_wait_seen:
            max_red_wait_seen = rw
        red_wait_history.append((step, sim.current_phase, sim.time_in_phase, list(sim.approach_red_wait), list(sim.queues)))
        
        if sim.current_phase == current_phase:
            current_dur += 1
        else:
            phase_durations.append((current_phase, current_dur))
            current_phase = sim.current_phase
            current_dur = 1
            
    print(f"Max red wait seen on int_0: {max_red_wait_seen:.1f}s")
    print("\nSample Phase Transition Durations (phase, seconds held):")
    for p, d in phase_durations[:12]:
        p_name = {0: "N-S Green", 1: "N-S Yellow", 2: "E-W Green", 3: "E-W Yellow"}[p]
        print(f"  {p_name} (phase {p}): held for {d}s")
        
    # Find the step where max red wait occurred and print its window
    max_step = max(range(len(red_wait_history)), key=lambda s: max(red_wait_history[s][3]))
    print(f"\nStep of peak red wait ({red_wait_history[max_step][3]}): step {max_step}")
    start = max(0, max_step - 25)
    print("Trace leading up to peak red wait:")
    for s in range(start, max_step + 1):
        step_idx, p, tip, rws, qs = red_wait_history[s]
        p_name = {0: "NS_G", 1: "NS_Y", 2: "EW_G", 3: "EW_Y"}[p]
        print(f"  Step {step_idx:3d} | Phase {p} ({p_name:4s}) tip={tip:2d} | Queues: {[round(q,1) for q in qs]} | RedWait: {[round(w,1) for w in rws]}")

if __name__ == "__main__":
    run_trace()
