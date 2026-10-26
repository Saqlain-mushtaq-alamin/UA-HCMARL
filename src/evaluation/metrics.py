"""
Metrics Suite (Phase 1 §1.5)

Computes:
- Primary Performance Metrics (Delay, Queue Length, Throughput, Travel Time)
- Constraint Metrics (Violation Rate, Violation Magnitude, Per-Type Count)
- Robustness Metrics (Degradation Slope, Failure Recovery Time)
- Fairness / Load-Balancing Metrics (Delay Variance, Max Delay, Jain's Fairness Index)
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import stats


class MetricsCalculator:
    """
    Tracks and computes all metrics across single or multi-agent TSC simulation runs.
    """

    def __init__(self, num_intersections: int = 1):
        self.num_intersections = num_intersections
        self.reset()

    def reset(self):
        """Reset step and episode history."""
        self.delays: List[Dict[str, float]] = []
        self.queue_lengths: List[Dict[str, float]] = []
        self.throughputs: List[int] = []
        self.travel_times: List[float] = []
        
        # Constraint tracking
        self.violations: List[Dict[str, int]] = []  # per step count of violations
        self.violations_per_agent: List[Dict[str, Dict[str, int]]] = []
        self.violation_magnitudes: List[float] = []
        self.violation_types: Dict[str, int] = {
            "green_time": 0,
            "green_skip": 0,
            "phase_skip": 0,
        }

    def record_step(
        self,
        intersection_delays: Dict[str, float],
        intersection_queues: Dict[str, float],
        step_throughput: float = 0.0,
        step_travel_time: Optional[float] = None,
        step_violations: Optional[Dict[str, int]] = None,
        step_violation_magnitude: float = 0.0,
        step_violations_per_agent: Optional[Dict[str, Dict[str, int]]] = None,
    ):
        """Record metrics for a single simulation step."""
        self.delays.append(intersection_delays)
        self.queue_lengths.append(intersection_queues)
        self.throughputs.append(float(step_throughput))
        
        if step_travel_time is not None and step_travel_time > 0:
            self.travel_times.append(step_travel_time)

        if step_violations:
            self.violations.append(step_violations)
            for vtype, count in step_violations.items():
                if vtype in self.violation_types:
                    self.violation_types[vtype] += count
                else:
                    self.violation_types[vtype] = count
        else:
            self.violations.append({})

        if step_violations_per_agent:
            self.violations_per_agent.append(step_violations_per_agent)

        self.violation_magnitudes.append(step_violation_magnitude)

    def compute_summary(self) -> Dict[str, float]:
        """Compute aggregated metrics summary for the episode."""
        if not self.delays:
            return {}

        # 1. Primary Metrics
        all_delays = [val for d in self.delays for val in d.values()]
        avg_delay = float(np.mean(all_delays)) if all_delays else 0.0

        all_queues = [val for q in self.queue_lengths for val in q.values()]
        avg_queue = float(np.mean(all_queues)) if all_queues else 0.0

        total_throughput = float(np.sum(self.throughputs))
        avg_travel_time = float(np.mean(self.travel_times)) if self.travel_times else 0.0

        # 2. Constraint Metrics (Per-Intersection & System-Level)
        total_steps = len(self.delays)
        num_intersections = len(self.delays[0]) if self.delays else 1
        total_agent_steps = max(1, total_steps * num_intersections)

        # True per-intersection violation rate: fraction of intersection-steps that violated
        if self.violations_per_agent:
            violating_agent_steps = sum(
                1 for step_v in self.violations_per_agent
                for a_v in step_v.values()
                if sum(a_v.values()) > 0
            )
            per_int_violation_rate = float(violating_agent_steps / total_agent_steps)
        else:
            # Fallback when per-agent breakdown is omitted: normalize total violations by agent-steps
            total_viol_events = sum(sum(v.values()) for v in self.violations)
            per_int_violation_rate = min(1.0, float(total_viol_events / total_agent_steps))

        # System-level violation rate (any intersection violated in that step)
        steps_with_violation = sum(1 for v in self.violations if sum(v.values()) > 0)
        system_violation_rate = float(steps_with_violation / total_steps) if total_steps > 0 else 0.0

        total_violation_magnitude = float(np.sum(self.violation_magnitudes))

        # Rates per violation type (per intersection-step)
        green_time_rate = float(self.violation_types.get("green_time", 0) / total_agent_steps)
        phase_skip_rate = float(self.violation_types.get("phase_skip", 0) / total_agent_steps)
        green_skip_rate = float(self.violation_types.get("green_skip", 0) / total_agent_steps)

        # 3. Fairness Metrics (Per intersection average delays)
        intersection_ids = list(self.delays[0].keys()) if self.delays else []
        per_int_delays = []
        for i_id in intersection_ids:
            i_delays = [step_d.get(i_id, 0.0) for step_d in self.delays]
            per_int_delays.append(np.mean(i_delays))

        if per_int_delays:
            per_int_arr = np.array(per_int_delays, dtype=np.float64)
            delay_variance = float(np.var(per_int_arr))
            max_intersection_delay = float(np.max(per_int_arr))
            
            # Jain's Fairness Index
            sum_d = np.sum(per_int_arr)
            sum_sq_d = np.sum(per_int_arr ** 2)
            n_int = len(per_int_arr)
            jains_fairness = float((sum_d ** 2) / (n_int * sum_sq_d + 1e-8)) if sum_sq_d > 0 else 1.0
        else:
            delay_variance = 0.0
            max_intersection_delay = 0.0
            jains_fairness = 1.0

        summary = {
            "avg_delay": avg_delay,
            "avg_queue_length": avg_queue,
            "total_throughput": float(total_throughput),
            "avg_travel_time": avg_travel_time,
            # Primary rate: Per-intersection violation rate (avoids 16-intersection saturation)
            "constraint_violation_rate": per_int_violation_rate,
            "per_intersection_violation_rate": per_int_violation_rate,
            "system_violation_rate": system_violation_rate,
            "total_violation_magnitude": total_violation_magnitude,
            "green_time_violations": float(self.violation_types.get("green_time", 0)),
            "green_skip_violations": float(self.violation_types.get("green_skip", 0)),
            "phase_skip_violations": float(self.violation_types.get("phase_skip", 0)),
            "green_time_rate": green_time_rate,
            "phase_skip_rate": phase_skip_rate,
            "green_skip_rate": green_skip_rate,
            "delay_variance": delay_variance,
            "max_intersection_delay": max_intersection_delay,
            "jains_fairness_index": jains_fairness,
        }
        return summary


def compute_robustness_degradation_slope(
    noise_levels: List[float], delays_per_level: List[float]
) -> Tuple[float, float, float]:
    """
    Computes linear regression slope of performance (delay) vs noise level.
    Lower slope = higher robustness to noise.
    Returns: (slope, intercept, r_value).
    """
    if len(noise_levels) < 2:
        return 0.0, 0.0, 0.0

    res = stats.linregress(noise_levels, delays_per_level)
    return float(res.slope), float(res.intercept), float(res.rvalue)


def compute_recovery_time(
    delays_over_time: List[float],
    onset_step: int,
    baseline_delay: float,
    tolerance: float = 0.1,
) -> int:
    """
    Computes number of steps required to return to baseline performance ± tolerance after failure onset.
    """
    threshold = baseline_delay * (1.0 + tolerance)
    for t in range(onset_step, len(delays_over_time)):
        if delays_over_time[t] <= threshold:
            return t - onset_step
    return len(delays_over_time) - onset_step
