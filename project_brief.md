# Project Brief: OffshoreWind_ControlRL

## One-Sentence Summary

Train reinforcement learning agents in a custom Gymnasium environment to stabilise a simplified floating offshore wind platform under stochastic wind and wave disturbances, demonstrating a complete RL control pipeline with baselines, safety filtering, and rigorous multi-seed robustness evaluation.

---

## Project Goal

The goal of **OffshoreWind_ControlRL** is to build a lightweight, original, and reproducible reinforcement learning control system that demonstrates how a physical engineering control problem — floating offshore wind turbine platform stabilisation — can be formulated as a Markov Decision Process (MDP) and solved using modern deep RL methods.

Specifically, the project:

1. **Builds** a custom Gymnasium environment simulating simplified single-axis pitch dynamics under wind/wave disturbances
2. **Trains** PPO controllers (Stable-Baselines3) with standard and domain-randomised protocols
3. **Compares** RL agents against classical baselines (NoControl, PD controller)
4. **Evaluates** robustness across four wind/wave scenarios including out-of-distribution conditions
5. **Implements** a model-based safety filter that modifies unsafe RL actions before execution
6. **Demonstrates** honest, reproducible science: multi-seed protocols, seed-variance analysis, and clear reporting of where RL does and does not add value

The implementation remains lightweight and self-contained — no OpenFAST, no FLORIS, no heavy simulators — making it suitable for a Deep Reinforcement Learning mini-project and a strong applied AI portfolio piece.

**Design principle**: Runs on CPU by default. The environment's tiny state space (4 dimensions) and SB3's default 2-layer MLP (~8K parameters) make CPU training fast and reproducible without GPU or Apple Silicon MPS.

---

## Verified Results Summary

A rigorous Round 3 evaluation (3 seeds, 500k timesteps, 4 scenarios, 20 episodes each) produced the following verified findings:

| Scenario | Winner | avg_return | Evidence Strength |
|----------|--------|-----------|-------------------|
| `normal_wind` | **PD** | -2.59 | Very strong — PD beats PPO by >4.6x, zero failures, 7x less energy |
| `strong_wind` | **PD** | -10.10 | Strong — PD beats best PPO by >3x, zero failures vs 20-30% |
| `variable_wind` | **PD** | -10.90 | Very strong — PD beats best PPO by >5x, zero failures vs 33% |
| `out_of_distribution` | **PPO+Safety** | -30.19 | Weak/narrow — 18% better than PD but higher failure rate, n=3 |

**Three key honest takeaways:**

1. **Classical PD remains the strongest and most reliable controller** in all in-distribution settings — zero failures, deterministic behaviour, and dramatically lower control energy (7x less in normal wind).
2. **PPO is high-variance and seed-dependent.** Seed 0 fails catastrophically (100% failure in variable/OOD wind) while seeds 1-2 can sometimes compete with PD. Single-seed evaluations are misleading.
3. **Safety-filtered PPO shows promise in OOD conditions** but with caveats: higher failure rate than PD, near-constant safety intervention, and limited statistical power (n=3 seeds).

Full verified results: `reports/round3_result_audit.md`.

---

## Target Audience

- **Portfolio reviewers** evaluating applied AI and engineering ML competence
- **Applied AI researchers** exploring RL for physical system control
- **Deep RL course instructors** looking for reproducible project examples with honest evaluation
- **PhD supervisors** assessing candidate capability in RL problem formulation and scientific methodology
- **Industrial Engineering / Operations Research readers** interested in AI for energy systems
- **Control system designers** curious about learning-based vs classical control approaches

---

## Technical Stack

| Component | Technology |
|-----------|-----------|
| Environment framework | Gymnasium |
| Numerics | NumPy |
| RL library | Stable-Baselines3 |
| Deep learning backend | PyTorch (CPU) |
| Data analysis | Pandas |
| Visualization | Matplotlib |
| Testing | pytest |
| Language | Python 3.10+ |

---

## Evidence Boundary

This project is inspired by public information and related open-source projects, but **the implementation is entirely original**.

### Sources of Inspiration

| Source | Concept Borrowed | This Project's Implementation |
|--------|-----------------|------------------------------|
| **DeepSense** | Deep learning applied to ocean/sensor data | Entire codebase independently written |
| **RL-PSF** | Concept of safety-aware RL / predictive safety filtering | Safety filter algorithm is independently designed |
| **FloatingFarmYaw** | RL for floating wind control | Environment, dynamics, and RL formulation are original |

### Key Principle

When discussing external projects in documentation:
- Clearly separate **confirmed public information** from **technical inference** and **this project's own implementation choices**
- No code is copied from any GitHub repository or proprietary source
- All design choices, hyperparameters, and reward weights are independently determined

---

## Completion Checklist

- [x] Custom Gymnasium environment (`FloatingPlatformEnv`)
- [x] NoControl baseline
- [x] PD Controller baseline
- [x] PPO training script with CLI arguments
- [x] Domain-randomized PPO training script
- [x] Safety filter with `SafetyFilteredController` wrapper
- [x] Evaluation script comparing all controllers
- [x] Metrics CSV output (evaluation_summary.csv + evaluation_summary_round3.csv)
- [x] Trajectory CSV output (per-scenario)
- [x] Result plotting script (6+ plot types including Round 3 figures)
- [x] README with reproduction instructions
- [x] Methodology document (`methodology.md`)
- [x] MVP report (`reports/mvp_report.md`)
- [x] Round 3 verification audit (`reports/round3_result_audit.md`)
- [x] Pytest smoke tests
- [x] Multi-seed evaluation protocol (Round 3)

---

## Resume Bullet Draft

> Designed and evaluated a reinforcement learning control pipeline for floating offshore wind platform stabilisation, training PPO agents across three random seeds at 500k timesteps. Multi-seed analysis revealed that classical PD control outperforms RL across 3 of 4 wind scenarios with zero failures and 7x lower actuator energy — a validated "no free lunch" finding that demonstrates honest scientific evaluation rather than forced RL superiority. Implemented domain randomisation, safety filtering, and a rigorous evaluation protocol, producing portfolio-grade evidence for when RL is and is not the right tool for control tasks.

---

## Disclaimer

> **This is a simplified simulation based reproduction for learning and portfolio purposes. It does not claim engineering fidelity to real floating offshore wind turbine systems.**
