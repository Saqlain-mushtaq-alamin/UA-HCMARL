# UA-HCMARL

**Uncertainty-Aware Hierarchical Constrained Multi-Agent Reinforcement Learning for Traffic Signal Control**

UA-HCMARL is a research framework for training and evaluating multi-agent reinforcement-learning controllers for urban traffic signals. It studies how traffic-signal policies behave when sensor observations are noisy or degraded.

The project combines:

1. **Belief-state estimation** - a GRU-KalmanNet-style encoder estimates a cleaner traffic state and its uncertainty.
2. **Constrained worker agents** - intersection-level policies use MAPPO-style learning with PID-Lagrangian constraint handling.
3. **Uncertainty-aware hierarchy** - regional managers allocate signal-control budgets to workers using the reported uncertainty.

The project also includes fixed-time, actuated, IPPO, MAPPO, MAPPO-LCE, and FMA2C baselines.

## What this project measures

Experiments evaluate controllers on:

- Average queue length (vehicles)
- Throughput
- Constraint-violation rate
- Robustness as sensor noise increases
- Statistical significance, including paired tests, Wilcoxon tests, and Cohen's *d*
- Belief-state calibration and uncertainty quality
- The effect of removing individual architecture components

> **Metric note:** Older script output sometimes labels queue length as “delay”. The environment and current reports treat this value as average queue length in vehicles, not seconds of travel-time delay.

## Research architecture

```text
Noisy detector observations
            |
            v
  Belief-State Encoder (BSE)
  estimated state + uncertainty
            |
            v
  Constrained worker policies
  one worker per intersection
            |
            v
  Regional manager policies
  uncertainty-aware budget allocation
            |
            v
     Traffic-signal actions
```

The main controller is implemented in [`src/baselines/ua_hcmarl.py`](src/baselines/ua_hcmarl.py). Its components are implemented in [`src/models/`](src/models/) and trained with the utilities in [`src/training/`](src/training/).

## Repository layout

| Path | Purpose |
| --- | --- |
| [`src/envs/`](src/envs/) | Traffic environments, signal dynamics, and sensor-noise injection |
| [`src/baselines/`](src/baselines/) | Baseline controllers and the proposed UA-HCMARL controller |
| [`src/models/`](src/models/) | Belief encoder, worker/manager models, Lagrangian multipliers, and budget projection |
| [`src/training/`](src/training/) | RL training, BSE pre-training, curriculum, and callbacks |
| [`src/evaluation/`](src/evaluation/) | Metrics, visualizations, and statistical analysis |
| [`scripts/`](scripts/) | Experiment runners, diagnostics, and figure-generation scripts |
| [`configs/`](configs/) | Environment, agent, noise, and experiment configuration files |
| [`networks/`](networks/) | SUMO network and route definitions |
| [`tests/`](tests/) | Unit tests for environments, models, baselines, noise, metrics, and statistics |
| [`experiments/`](experiments/) | Current result files, figures, and trained checkpoints |
| [`planning/`](planning/) | Research design, phase descriptions, and execution guidance |

## Requirements

The project targets Python 3.11 or a compatible modern Python 3 release. Install the runtime and test dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy matplotlib torch gymnasium pytest
```

The environment can run with the built-in Python traffic simulator. SUMO/TraCI is optional; when `traci` and `sumolib` are unavailable, [`src/envs/sumo_env.py`](src/envs/sumo_env.py) uses its native fallback simulator.

If you use SUMO for external network simulation, install SUMO separately and make its tools available on `PATH`.

## Quick start

Run the unit tests first:

```powershell
python -m pytest -q
```

Run a short end-to-end validation:

```powershell
python scripts/run_phase2_benchmark.py --profile smoke
python scripts/run_phase3_benchmark.py --profile smoke
python scripts/run_phase5_ablation_benchmark.py --profile smoke
```

The smoke profile uses a small network, two seeds, short episodes, and a small training budget. It is intended to verify the pipeline, not to produce publishable results.

## Experiment profiles

All benchmark runners support the profiles defined in [`experiment_config.py`](experiment_config.py):

| Profile | Intersections | Seeds | Training episodes | Steps/episode | Use |
| --- | ---: | ---: | ---: | ---: | --- |
| `smoke` | 4 | 2 | 2 | 100 | Pipeline and code validation |
| `fast` | 16 | 5 | 10 | 500 | Development and quick comparisons |
| `canonical` | 16 | 10 | 200 | 3600 | Publication-oriented benchmark |

Noise scenarios are `clean`, `mild`, `moderate`, `severe`, and `extreme`, corresponding to noise levels `0.0`, `0.1`, `0.2`, `0.3`, and `0.5`.

## Running the research phases

### Phase 1 - classical and basic baselines

Run baseline experiments with:

```powershell
python scripts/run_baselines.py
```

Results are written under `experiments/phase1_baselines/`.

### Phase 2 - flat baseline benchmark

This compares Fixed-Time, Actuated, IPPO, MAPPO, MAPPO-LCE, and FMA2C:

```powershell
python scripts/run_phase2_benchmark.py --profile canonical
```

For a faster run:

```powershell
python scripts/run_phase2_benchmark.py --profile fast --num-workers 4
```

Useful options include `--skip-train`, `--force-train`, `--force-eval`, `--no-parallel-train`, and `--device cuda`.

### Phase 3 - belief-state encoder

This phase pre-trains and evaluates the belief-state encoder, then compares confidence modulation with an ablation that disables it:

```powershell
python scripts/run_phase3_benchmark.py --profile canonical
```

### Phase 4 - full hierarchical system

This phase evaluates UA-HCMARL against the baselines and analyzes uncertainty-aware budget reallocation:

```powershell
python scripts/run_phase4_benchmark.py --profile canonical
```

### Phase 5 - controlled ablations

This phase isolates the contribution of the main components:

```powershell
python scripts/run_phase5_ablation_benchmark.py --profile canonical
```

The ablations include:

- **A0:** Full UA-HCMARL
- **A1:** No hierarchy
- **A2:** No belief state
- **A3:** No uncertainty coupling
- **A4:** No constraints
- **A5:** Flat constrained baseline
- **A6:** Flat unconstrained baseline

### Run the complete pipeline

The legacy all-in-one runner is also available:

```powershell
python scripts/run_full_experiment.py
python scripts/run_full_experiment.py --fast
```

For the phase-specific publication workflow, prefer the profile-based runners above because they share the centralized configuration and checkpointing behavior.

## Checkpointing and resuming

Long-running evaluations use [`checkpointed_runner.py`](checkpointed_runner.py). Results are saved atomically after each completed scenario, controller, and seed combination.

If a run is interrupted, rerun the same command. Existing completed evaluations are detected and skipped, allowing the benchmark to continue without discarding earlier results.

For overnight runs, redirect output to a log:

```powershell
$p = Start-Process python `
  -ArgumentList "scripts/run_phase2_benchmark.py --profile canonical" `
  -NoNewWindow -PassThru `
  -RedirectStandardOutput "phase2_canonical.log" `
  -RedirectStandardError "phase2_canonical_err.log"

Get-Content phase2_canonical.log -Tail 25 -Wait
```

## Results and generated artifacts

Benchmark runners normally write JSON summaries, figures, model checkpoints, learning curves, and statistical reports to:

- `experiments/phase1_baselines/`
- `experiments/phase2_flat_baselines/`
- `experiments/phase3_belief_encoder/`
- `experiments/phase4_full_hierarchy/`
- `experiments/phase5_ablations/`

Existing results can be inspected with:

```powershell
python scripts/inspect_results.py
python scripts/generate_figures.py
```

The [`planning/`](planning/) directory contains the research proposal, phase definitions, result summaries, and the detailed restart guide.

## Reproducibility notes

- Use the same profile, network, route file, and seed count when comparing methods.
- Use `canonical` for publication-oriented numbers; use `smoke` only for validation.
- Do not compare results produced with different episode horizons or training budgets without reporting the difference.
- Preserve the generated JSON files alongside figures so statistical results can be reproduced.
- The default network is the 4x4 grid in [`networks/grid4x4/`](networks/grid4x4/). A smaller Lemgo network is also available in [`networks/lemgorl/`](networks/lemgorl/).

## Current project status

The repository contains implemented environments, baselines, the proposed hierarchical controller, evaluation utilities, tests, trained checkpoints, and phase result artifacts. The canonical benchmark is computationally expensive; run the smoke profile before starting a full experiment.

For the most detailed execution instructions, see [`planning/EXPERIMENT_EXECUTION_AND_RESTART_GUIDE.md`](planning/EXPERIMENT_EXECUTION_AND_RESTART_GUIDE.md). For the research motivation and novelty statement, see [`planning/overview/RL_TSC_Research_Proposal_Summary.md`](planning/overview/RL_TSC_Research_Proposal_Summary.md).
