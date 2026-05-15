# OffshoreWind_ControlRL

**Reinforcement Learning for Control of Floating Offshore Wind Platforms**

> **Disclaimer**: This is a simplified simulation-based reproduction for learning and portfolio purposes. It does not claim engineering fidelity to real floating offshore wind turbine systems.

---

## Final Results Summary

A rigorous multi-seed evaluation (3 seeds, 500k timesteps, 4 wind scenarios, 20 episodes each) compared classical PD control against PPO-based reinforcement learning controllers across six configurations.

| Scenario | Winner | avg_return | Failure Rate | Notes |
|----------|--------|-----------|-------------|-------|
| `normal_wind` | **PD** | **-2.59** | 0.0% | PD beats all PPO variants by >4.6x return margin, with 7x less control energy |
| `strong_wind` | **PD** | **-10.10** | 0.0% | PD dominates; best PPO variant averages 3x worse with 20-30% failure |
| `variable_wind` | **PD** | **-10.90** | 0.0% | PD clear winner; all PPO variants cluster at 5x worse returns with 33% failure |
| `out_of_distribution` | **PPO+Safety** | **-30.19** | 40.0% | 18% return improvement over PD (-36.78), but with higher failure rate (40% vs 35%) and n=3 seeds |

**Key takeaways:**

- **Classical PD remains the strongest and most reliable controller** in all in-distribution settings — zero failures, deterministic behaviour, and dramatically lower control energy.
- **PPO is high-variance and seed-dependent.** Seed 0 fails catastrophically across all PPO variants in variable and OOD wind (100% failure rate), while seeds 1-2 can sometimes compete with or exceed PD. The bimodal distribution makes single-seed evaluation misleading.
- **Safety-filtered PPO shows promise in OOD conditions**, narrowly winning the out-of-distribution scenario, but with caveats: higher failure rate than PD, near-constant safety intervention (93.5% intervention rate), and limited statistical power (n=3 seeds).
- **Domain randomization has mixed results** — it helps slightly in strong wind and OOD but worsens performance in variable wind, and does not solve the seed-0 failure problem.

Full verified results are in `reports/round3_result_audit.md`.

---

## Overview

This project applies reinforcement learning (PPO) to stabilise a simplified floating offshore wind platform exposed to stochastic wind and wave disturbances. It demonstrates a complete RL control pipeline:

1. Custom Gymnasium environment with simplified single-axis pitch dynamics
2. Classical baselines (NoControl, PD controller)
3. PPO training with domain randomisation option
4. Model-based safety filter for action constraint enforcement
5. Robustness evaluation across 4 wind/wave scenarios (normal, strong, variable, OOD)
6. Multi-seed evaluation protocol (3 seeds, 500k timesteps) to ensure statistical reliability

**Target Audience**: Portfolio reviewers, applied AI researchers, deep RL course instructors, and anyone interested in RL for physical system control.

---

## Quick Start

### Installation

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Smoke Test

```bash
# Run pytest smoke tests for the environment
pytest tests/

# Or run the environment's built-in self-test
python -m src.envs.floating_platform_env
```

### Quick Training Smoke Test (1,000 timesteps)

```bash
python -m src.training.train_ppo --smoke-test --scenario normal_wind
```

### 50k Training — Quick Experimentation

```bash
# Standard PPO
python -m src.training.train_ppo --timesteps 50000 --seed 42 --scenario normal_wind

# Other scenarios
python -m src.training.train_ppo --timesteps 50000 --scenario strong_wind
python -m src.training.train_ppo --timesteps 50000 --scenario variable_wind
```

### 500k Training — Reproducing Round 3 Results

```bash
# Train PPO with 3 seeds for a single scenario
python -m src.training.train_ppo --timesteps 500000 --seed 0 --scenario normal_wind
python -m src.training.train_ppo --timesteps 500000 --seed 1 --scenario normal_wind
python -m src.training.train_ppo --timesteps 500000 --seed 2 --scenario normal_wind

# Domain-randomised PPO
python -m src.training.train_randomized_ppo --timesteps 500000 --seed 0 --scenario normal_wind
python -m src.training.train_randomized_ppo --timesteps 500000 --seed 1 --scenario normal_wind
python -m src.training.train_randomized_ppo --timesteps 500000 --seed 2 --scenario normal_wind

# Repeat for strong_wind, variable_wind, out_of_distribution_wind
```

### Evaluate All Controllers

```bash
# Evaluates NoControl, PD, PPO, and PPO+Safety across all 4 scenarios
# Requires a trained model (defaults to results/models/ppo_normal_wind.zip)
python -m src.evaluation.evaluate_agents --episodes 20

# Target a specific model
python -m src.evaluation.evaluate_agents --model results/models/ppo_strong_wind.zip --episodes 20
```

### Generate Plots

```bash
# Reads evaluation_summary.csv and trajectory CSVs, produces PNG figures
python -m src.analysis.plot_results
```

---

## Project Structure

```
OffshoreWind_ControlRL/
  README.md                           # This file
  project_brief.md                    # Project description and portfolio positioning
  methodology.md                      # MDP formulation, dynamics, baselines, safety logic
  requirements.txt                    # Python dependencies

  implementation_plans/
    01_mvp_implementation_plan.md

  src/
    envs/
      floating_platform_env.py        # Custom Gymnasium environment (4-dim state, 1-dim action)
    baselines/
      no_control.py                   # Zero-action baseline controller
      pd_controller.py                # PD feedback controller (Kp=5.0, Kd=2.0)
    safety/
      simple_safety_filter.py         # One-step-ahead predictive safety filter
    training/
      train_ppo.py                    # Standard PPO training (MlpPolicy)
      train_randomized_ppo.py         # Domain-randomised PPO training
    evaluation/
      evaluate_agents.py              # Multi-controller evaluation pipeline
    analysis/
      plot_results.py                 # Result visualisation (6 plot types)

  results/
    models/                           # Saved PPO .zip weights + checkpoint snapshots
    logs/                             # Training CSV logs
    metrics/                          # evaluation_summary.csv + per-scenario trajectory CSVs
    figures/                          # Generated PNG plots

  reports/
    mvp_report.md                     # MVP results and limitations
    round3_result_audit.md            # Verified Round 3 findings with per-seed analysis

  tests/
    test_env_smoke.py                 # Pytest smoke tests for the environment
```

---

## Available Scenarios

| Scenario | wind_std | wave_std | Description |
|----------|----------|----------|-------------|
| `normal_wind` | 0.3 | 0.2 | Baseline operating conditions |
| `strong_wind` | 0.6 | 0.4 | Elevated wind and wave activity |
| `variable_wind` | 0.3 | 0.2 | Wind mean ramps from 0 to 0.5 over the episode |
| `out_of_distribution_wind` | 0.9 | 0.7 | Extreme disturbance levels for robustness testing |

---

## How to Interpret Results

### Metrics

| Metric | What It Measures | Good = |
|--------|-----------------|--------|
| `avg_return` | Average cumulative reward per episode | Higher (less negative) |
| `mean_abs_theta` | Average pitch deviation in radians | Lower |
| `max_abs_theta` | Worst-case absolute pitch angle observed | Lower (must stay < 0.3 rad) |
| `failure_rate` | Fraction of episodes hitting the safety limit (|theta| > 0.3) | Lower (0 is best) |
| `control_energy` | Mean sum-of-squared actions per episode — proxy for actuator wear | Lower (context-dependent) |

### Controllers Compared

| Controller | Description |
|------------|-------------|
| **NoControl** | Always returns `a = 0`. Reveals uncontrolled dynamics — worst baseline. |
| **PD** | Classical proportional-derivative feedback (`action = clip(-Kp*theta - Kd*theta_dot)` with Kp=5.0, Kd=2.0). Provides a well-understood linear-control reference. |
| **PPO** | Proximal Policy Optimisation via Stable-Baselines3 MlpPolicy (lr=3e-4, gamma=0.99, 2-layer-64-hidden). Learns to counteract disturbances. |
| **PPO+Safety** | PPO wrapped in a one-step-ahead model predictive safety filter. Predicts next-step pitch; if |theta_predicted| > 0.3 rad, emergency stop (action=0); if > 0.24 rad, halve the action. |
| **PPO-Randomized** | PPO trained with domain randomisation (physics parameters vary per episode reset). |
| **PPO-Randomized+Safety** | Domain-randomized PPO + safety filter wrapper. |

---

## RL Formulation (Summary)

- **State (observation)**: `[theta, theta_dot, wind_disturbance, wave_disturbance]` — 4 dimensions, R^4
- **Action**: Single continuous value a in [-1, 1], mapped to control force `f = a * action_gain` (default gain = 0.5 N·m)
- **Dynamics**: Forward Euler integration of a mass-spring-damper model:
  ```
  theta_next = theta + theta_dot * dt
  theta_dot_next = theta_dot + (dt/mass) * (wind + wave + f_control - damping*theta_dot - stiffness*theta)
  ```
- **Reward**: `r = -1.0*theta^2 - 0.5*theta_dot^2 - 0.1*f_control^2 + safety_penalty` (safety_penalty = -10.0 when |theta| > 0.9 * 0.3)
- **Done**: `terminated` if |theta| > 0.3 rad; `truncated` at `max_steps = 1000` (50 seconds of simulated time)
- **Algorithm**: PPO via Stable-Baselines3 (MlpPolicy, default hyperparameters)

Full mathematical details are in [methodology.md](methodology.md).

---

## Round 3: Multi-Seed Evaluation and Extended PPO Training (COMPLETE)

Round 3 addresses two methodological concerns from Rounds 1 and 2: (1) single-seed evaluation makes it impossible to distinguish genuine policy improvements from random luck, and (2) 50,000 training timesteps may still be insufficient for PPO to converge to a competitive policy.

### Protocol

- **Seeds**: 3 (0, 1, 2)
- **Training**: 500,000 timesteps per seed per controller type
- **Controllers**: NoControl, PD, PPO (500k), PPO+Safety (500k), PPO-Randomized (500k), PPO-Randomized+Safety (500k)
- **Scenarios**: All four (normal_wind, strong_wind, variable_wind, out_of_distribution_wind)
- **Episodes**: 20 per evaluation
- **Metrics**: mean +/- std across seeds for avg_return, failure_rate, mean_abs_theta, control_energy

### 3.3 Verified Results

| Scenario | Winner | Evidence Quality |
|----------|--------|-----------------|
| normal_wind | **PD** (-2.59) | Very strong — PD beats all PPO variants by >4.6x margin, zero failures, lowest energy |
| strong_wind | **PD** (-10.10) | Strong — PD beats best PPO variant by >3x margin, zero failures vs 20-30% |
| variable_wind | **PD** (-10.90) | Very strong — PD beats best PPO by >5x margin, zero failures vs 33% |
| OOD_wind | **PPO+Safety** (-30.19) | Weak/narrow — beats PD by 18%, but at higher failure rate (40% vs 35%) with only n=3 seeds |

**PD Performance Summary:**

| Metric | normal_wind | strong_wind | variable_wind | OOD_wind |
|--------|-------------|-------------|--------------|----------|
| avg_return | -2.59 | -10.10 | -10.90 | -36.78 |
| failure_rate | 0.0% | 0.0% | 0.0% | 35.0% |
| control_energy | 25.10 | 98.80 | 154.35 | 189.58 |

**PPO Seed-Variance Issue:**

Seed 0 fails catastrophically across all PPO variants in variable_wind and OOD_wind (100% failure rate). Seeds 1-2 are sometimes competitive with PD. This bimodal distribution makes single-seed PPO conclusions unreliable and highlights a fundamental deployment risk.

**PPO+Safety in OOD:**

The only scenario where a PPO variant wins. avg_return of -30.19 beats PD's -36.78 (18% improvement). However, this comes with a higher failure rate (40.0% vs 35.0%) and requires near-constant safety intervention, making the practical value unclear.

### 3.4 Generated Figures

| Figure | Path |
|--------|------|
| Average return with error bars | `results/figures/round3_average_return_mean_std.png` |
| Failure rate comparison | `results/figures/round3_failure_rate_mean_std.png` |
| Mean absolute pitch | `results/figures/round3_mean_abs_pitch_mean_std.png` |
| Control energy | `results/figures/round3_control_energy_mean_std.png` |
| Robustness gap | `results/figures/round3_robustness_gap.png` |
| Learning curves by seed | `results/figures/round3_learning_curves_by_seed.png` |

---

## Evidence Boundary

This project is inspired by public research efforts including DeepSense, RL-PSF, and FloatingFarmYaw. **However, all implementation code is independently and originally written.** No code was copied from any external repository.

Key inspirations vs. original work:
- **DeepSense**: General concept of deep learning for ocean/sensor data — all code is original
- **RL-PSF**: Safety-aware RL concept — safety filter is independently designed
- **FloatingFarmYaw**: RL for floating wind concept — environment and RL formulation are original

See [methodology.md](methodology.md) for the full evidence boundary statement.

---

## Resume Bullet & Interview Notes

### Resume Bullet

> Designed and evaluated a reinforcement learning control pipeline for floating offshore wind platform stabilisation, training PPO agents across three random seeds at 500k timesteps. Multi-seed analysis revealed that classical PD control outperforms RL across 3 of 4 wind scenarios with zero failures and 7x lower actuator energy — a validated "no free lunch" finding that demonstrates honest scientific evaluation rather than forced RL superiority. Implemented domain randomisation, safety filtering, and a complete evaluation protocol, producing portfolio-grade evidence for when RL is and is not the right tool for control tasks.

### Interview Explanation

When asked about this project, the narrative is:

1. **The setup**: Built a custom Gymnasium environment modelling single-axis pitch dynamics of a floating wind platform under stochastic wind/wave disturbances. Trained PPO agents and compared them against classical PD control.

2. **The honest finding**: PD won decisively in 3 of 4 scenarios. This is not a failure — it's a scientifically important result. For a low-dimensional linear-ish system, PD is analytically close to optimal. RL's value is in high-dimensional, nonlinear, or multi-objective settings.

3. **The seed problem**: The most interesting finding was PPO's extreme seed variance. With 3 seeds, we found seed 0 consistently fails catastrophically while seeds 1-2 can compete with PD. This means single-seed evaluations are misleading — a critical lesson about RL reproducibility.

4. **The OOD nuance**: A safety-filtered PPO variant narrowly beat PD in the out-of-distribution scenario (18% improved return), but at a higher failure rate and with near-constant safety intervention. The margin is narrow and may not generalise — honest reporting matters.

5. **The takeaway**: This project demonstrates the ability to set up rigorous multi-seed evaluation, interpret results honestly (including negative findings), and articulate when classical methods are superior to learning-based approaches.

---

## Citation Note

If you reference this project, please cite it as a portfolio demonstration:

```
OffshoreWind_ControlRL: A simplified RL control system for floating offshore wind platforms.
Portfolio project for demonstrating deep reinforcement learning in applied AI.
Implementation is original and inspired by public research on floating wind control.
```

---

## License

MIT License

---

## Disclaimer

> This is a simplified simulation based reproduction for learning and portfolio purposes. It does not claim engineering fidelity to real floating offshore wind turbine systems. The simulation uses an abstract mass-spring-damper model with Gaussian noise disturbances — it is not a realistic offshore wind engineering simulator and should not be used for engineering design, safety certification, or operational decision-making.

## Known Modeling Limitations

- The observation space uses broad/unbounded bounds (Box(-inf, inf)), which does not reflect physical constraints.
- Reward coefficients are hand-tuned and not derived from a formal optimization process.
- The safety penalty is heuristic and discontinuous, applied as a fixed bonus when exceeding 90% of the safety threshold.
- PD and NoControl baselines are evaluated with a single seed (n=1), while PPO variants are evaluated with three seeds (n=3). This difference in seeding strategy should be considered when comparing variance.
