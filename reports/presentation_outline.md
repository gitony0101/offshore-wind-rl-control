# OffshoreWind_ControlRL — Course Presentation Outline

> RL vs. Classical Controllers for Floating Offshore Wind Turbines
> 9 slides with speaker notes

---

## Slide 1: Problem and Motivation

**Title:** Can Reinforcement Learning Control Floating Offshore Wind Turbines Better Than Classical Methods?

**Content bullets:**
- Floating offshore wind turbines face complex aerodynamic and hydrodynamic disturbances
- Maintaining rated rotor speed while minimizing platform pitch is a high-stakes control problem
- Classical PID/PD controllers are the industry standard — but RL promises adaptability
- **Core question:** Does a learned PPO policy actually beat a well-tuned PD controller on a simplified wind turbine environment?

**Speaker notes:**
Open with the big picture. Floating offshore wind is one of the fastest-growing renewable energy sectors, but controlling these turbines is genuinely hard. They have more degrees of freedom than fixed-bottom turbines, and the floating platform introduces coupling between rotor dynamics and wave motion. The industry standard solution is a gain-scheduled PI/PD controller — it works, but it's hand-tuned and doesn't adapt. Reinforcement learning has seen remarkable success in games, robotics, and even nuclear fusion plasma control. So the natural question is: can RL learn a better controller? That's what this project investigates — not with a toy problem, but with a physics-based floating wind turbine simulator. The answer, as we'll see, is more nuanced than most RL papers would have you believe.

**Suggested figure:** `results/figures/wind_wave_disturbance.png` — shows the environmental conditions the controller must handle.

---

## Slide 2: Why Floating Offshore Wind Control Matters

**Title:** Stakes Are High: Turbine Dynamics and Control Challenges

**Content bullets:**
- **Goal:** Maintain rated rotor speed (~12.1 RPM) despite wind and wave disturbances
- **Key challenge:** Platform pitch couples into rotor dynamics — unstable pitch causes structural damage
- **Actuator:** Blade pitch angle is the primary control; generator torque is secondary
- **Failure modes:** Rotor overspeed, underspeed, extreme platform pitch
- **Real-world stakes:** A failed controller means lost energy, structural fatigue, or turbine shutdown

**Speaker notes:**
Before diving into methods, let's understand why this is a real engineering problem and not just another gym environment. A floating wind turbine has to do two contradictory things simultaneously: extract maximum power from the wind AND keep the floating platform stable. These objectives conflict — aggressive pitch control to track rotor speed can destabilize the platform, while over-smoothing pitch loses energy revenue. The platform pitch is a slow, heavily coupled dynamic driven by waves that the controller can't directly observe. Failure means either overspeeding the rotor (mechanical damage), underspeeding (lost revenue), or pitching too far (capsizing risk). This is exactly the kind of multi-objective, partially-observable, safety-critical problem where RL theoretically should add value — but theoretically is the key word.

**Suggested figure:** `results/figures/control_action_comparison.png` — shows how different controllers move the pitch actuator over time.

---

## Slide 3: MDP Formulation

**Title:** Casting Turbine Control as a Reinforcement Learning Problem

**Content bullets:**
- **State (observation):** Rotor speed error, platform pitch, platform pitch rate, wind speed disturbance, and generator torque
- **Action:** Blade pitch angle adjustment (continuous, bounded to ±6 degrees from trim)
- **Reward function shaped to penalize:**
  - Rotor speed deviation from rated (primary term)
  - Platform pitch angle (structural stability)
  - Platform pitch rate (damping)
  - Control effort / actuator wear (energy penalty)
  - Failure terminal penalty (overspeed, underspeed, extreme pitch)
- **Episode termination (done):** Rotor speed ±20% from rated, platform pitch exceeds limits, or max episode length reached
- **Discount factor:** gamma = 0.99

**Speaker notes:**
The MDP formulation is where the physics meets the RL algorithm. The state contains the minimum set of quantities needed to make a control decision — rotor speed error tells us if we're at target, platform pitch and rate capture the floating platform stability, and wind speed disturbance gives a preview of incoming perturbations. Actions are continuous blade pitch adjustments, clipped to ±6 degrees from the trim operating point. The reward function is carefully shaped — the dominant term is rotor speed tracking error, but significant penalties on platform pitch and pitch rate prevent the policy from achieving good rotor tracking at the cost of platform instability. There's also a small control effort penalty to discourage jittery actuator movements that would cause mechanical wear. Episodes terminate when the controller loses control — typically the rotor spins too fast under strong wind gusts or the platform pitches too dramatically. These termination penalties are large (-100) to strongly discourage failure trajectories.

**Suggested figure:** `results/figures/learning_curve.png` — learning curves showing reward progression during training, illustrating the MDP in action.

---

## Slide 4: Environment and Methods

**Title:** Simplified Physics Model and Controller Baselines

**Content bullets:**
- **Environment:** Reduced-order floating wind turbine model (2-DOF: rotor speed + platform pitch)
- **Disturbances:** Time-varying wind speed, wave excitation forces, turbulence (Kaimal spectrum)
- **Controllers tested:**
  - **NoControl:** Baseline; pitch held at trim, no active control
  - **PD controller:** Classical proportional-derivative on rotor speed error (deterministic, hand-tuned)
  - **PPO:** Proximal Policy Optimization (3 random seeds)
  - **PPO + Safety:** PPO with intervention layer that overrides extreme pitch actions
  - **PPO-Randomized:** Domain-randomized training (varying wind/wave parameters)
  - **PPO-Randomized + Safety:** Domain randomization + safety intervention layer
- **PPO hyperparams:** Standard Stable-Baselines3 defaults, tuned for continuous action

**Speaker notes:**
We use a reduced-order model with two degrees of freedom — rotor speed and platform pitch. This simplification is deliberate: it isolates the core control coupling without the computational cost of a full finite-element model. The wind and wave disturbances are realistic — time-varying wind profiles, wave excitation forces, and Kaimal-spectrum turbulence. We test six controller types: a no-control baseline to establish the floor, a hand-tuned PD controller as the industry-standard bar, and four PPO variants. The vanilla PPO uses Stable-Baselines3 defaults. The +Safety variants add a heuristic intervention layer that overrides the policy when the pitch angle approaches physical limits. The -Randomized variants use domain randomization during training, perturbing wind and wave parameters to encourage robustness. Each PPO variant is trained with 3 random seeds to capture variance. The PD controller is our reference — it's deterministic, zero-variance, takes essentially no compute to evaluate, and represents decades of wind industry engineering.

**Suggested figure:** `results/figures/round3_robustness_gap.png` — shows how performance degrades from normal to OOD wind across controllers.

---

## Slide 5: Training Design

**Title:** Experimental Setup: 3 Seeds, 500K Steps, 4 Scenarios

**Content bullets:**
- **Training budget:** 500,000 timesteps × 3 random seeds per PPO variant
- **4 evaluation scenarios (increasing difficulty):**
  1. **normal_wind:** Benign conditions, mean wind ~11.4 m/s, small waves
  2. **strong_wind:** Above-rated wind ~18 m/s, larger waves — tests controller under stress
  3. **variable_wind:** Time-varying wind profile with turbulence — hardest in-distribution scenario
  4. **out_of_distribution_wind:** Extreme conditions outside training distribution
- **Evaluation protocol:** 20 episodes per controller × scenario combination (80 episodes per controller)
- **Total evaluations:** 15 controller/seed configs × 80 episodes = 1,200 evaluation episodes
- **Metrics:** Average return, failure rate, control energy, mean absolute pitch angle

**Speaker notes:**
The experimental design is structured to stress test each controller progressively. Training uses 500K steps per seed — enough time for PPO to converge on this environment, but deliberately not unlimited compute. We evaluate on four scenarios that increase in difficulty. Normal wind is the easy case — below the controller's worst expectations. Strong wind pushes the turbine into above-rated operation where blade pitch control becomes critical. Variable wind is the hardest in-distribution test — the wind speed changes over time with turbulence, so the controller can't learn a static response. Out-of-distribution wind tests generalization — the wind statistics differ significantly from anything seen during training. Each controller is evaluated for 20 episodes per scenario, giving us 80 data points per controller. We report four key metrics: average return for overall performance, failure rate for reliability, control energy for actuator cost, and mean absolute pitch angle for physical reasonableness. The use of 3 seeds per variant is a known limitation I'll discuss later.

**Suggested figure:** `results/figures/round3_learning_curves_by_seed.png` — learning curves separated by seed, showing training progression and seed-to-seed variance.

---

## Slide 6: Round 3 Results — The Honest Picture

**Title:** Results: PD Dominates 3 of 4 Scenarios; PPO Narrowly Wins OOD

**Content bullets:**
- **normal_wind:** PD wins decisively (return -2.59 vs PPO best -2.17 at single seed; mean PPO -14.49)
- **strong_wind:** PD wins by >5x margin (return -10.10 vs best PPO -37.13); PD has zero failures, PPO 21-30%
- **variable_wind:** PD wins by >5x margin (return -10.90 vs best PPO -58.73); PD zero failures, PPO 33% across all variants
- **OOD_wind:** PPO+Safety narrowly wins (-30.19 vs PD -36.78, 18% better) — but at 40% failure rate vs 35% for PD
- **Control energy:** PD uses 25 units in normal wind vs 186+ for PPO (~7x less); PD wins energy in 3 of 4 scenarios
- **PD standard deviation = 0** across all scenarios (deterministic controller)

**Speaker notes:**
Here's where the results get interesting — and maybe surprising if you came in expecting RL to dominate. PD wins three out of four scenarios. Not narrowly — decisively. In normal wind, PD achieves -2.59 return while the mean PPO is at -14.49, nearly 5x worse. In strong wind and variable wind, the gap widens to over 5x. PD also has a zero failure rate in all three in-distribution scenarios. In the OOD scenario, PPO+Safety achieves -30.19 versus PD's -36.78 — an 18% improvement. But there's a catch: PPO+Safety fails 40% of the time versus 35% for PD, so the margin on return comes at the cost of reliability. The energy numbers tell another story: PD uses 7x less control energy than PPO in normal wind. This matters because control energy translates to actuator wear and maintenance costs. PD's standard deviation is zero — it's a deterministic algorithm with fixed gains. There is no seed variance, no training instability, no question of whether it'll work when you deploy it.

**Suggested figure:** `results/figures/round3_average_return_mean_std.png` — bar chart of average returns with error bars across controllers and scenarios.

---

## Slide 7: Key Insight — The Seed Variance Problem

**Title:** The Bimodal Truth: Seed 0 Fails Catastrophically for Every PPO Variant

**Content bullets:**
- Across all PPO variants, **seed 0 consistently fails** — 100% failure rate in variable_wind and OOD_wind
- Seeds 1 and 2 often match or exceed PD performance (e.g., PPO-Randomized seed 1: -7.80 in strong_wind vs PD -10.10)
- **Result is bimodal, not random:** good seeds (1,2) cluster together, seed 0 is an outlier in the wrong direction
- Seed-to-seed return ranges are massive: up to 185 points in variable_wind (PPO+Safety: -196.60 to -11.65)
- Averaging across 3 seeds **hides** this bimodal distribution and produces a misleading "middle" number
- PD has zero seed variance — it works the same way every single time

**Speaker notes:**
This is the most important finding and the one I'm least comfortable reporting, but it's the most honest one. The PPO policies exhibit a bimodal performance distribution that averaging completely obscures. Seed 0 — the first random initialization — fails catastrophically in every single PPO variant. In variable_wind and OOD_wind, seed 0 has a 100% failure rate across all four PPO variants. Meanwhile, seeds 1 and 2 frequently achieve returns that match or beat PD. If you look at PPO-Randomized seed 1 in strong wind, it gets -7.80 — actually better than PD's -10.10. But you can't deploy a controller that works 2 out of 3 times and fails catastrophically on the third. The return ranges are enormous: for PPO+Safety in variable_wind, the best seed gets -11.65 and the worst gets -196.60 — that's a 185-point spread. With only 3 seeds, the average is meaningless because you're averaging a success with a failure. PD, by contrast, has zero variance. You tune it once, it works forever. This seed sensitivity is the single biggest barrier to deploying RL in safety-critical engineering applications.

**Suggested figure:** `results/figures/round3_failure_rate_mean_std.png` — failure rate bar chart showing the dramatic failure rates of PPO variants versus PD and NoControl.

---

## Slide 8: Limitations and Future Work

**Title:** What This Study Shows — and What It Doesn't

**Content bullets:**
- **Honest limitations:**
  - Only 3 seeds per variant — insufficient to characterize RL performance distribution (need n ≥ 30)
  - Simplified 2-DOF physics model, not a full aero-servo-elastic simulation
  - Single training budget (500K steps) — longer training might improve results
  - Safety layer is heuristic and adds intervention overhead without being learned
  - Domain randomization has mixed results (strong_wind improved, variable_wind worsened)
- **Future work:**
  - Test with 30+ seeds to properly quantify RL variance and confidence intervals
  - Investigate root cause of seed 0 failures (initialization vs. exploration vs. reward shaping)
  - Hybrid approaches: PD as safety fallback with RL as primary controller
  - Transfer learning: fine-tune PPO on harder scenarios starting from normal_wind policy
  - Full NREL 5MW turbine model with realistic wind/wave spectrum

**Speaker notes:**
I want to be explicit about what this study does and doesn't tell us. First, 3 seeds is not enough. It's better than the 1 seed most papers use, but it hides the bimodal distribution we're seeing. The proper approach would be 30+ seeds with confidence intervals. Second, our physics model is intentionally simplified — a 2-DOF model rather than a full 17-DOF aero-servo-elastic simulation from NREL. The results might change with more realistic dynamics, though I suspect the PD dominance would persist since PD is designed specifically for this kind of SISO tracking problem. Third, the safety layer is a rule-based intervention, not a learned constraint. It adds overhead and doesn't improve PPO's reliability meaningfully. Domain randomization shows mixed results — it helps in strong wind but hurts in variable wind. This makes sense: randomizing parameters can hurt performance when the real environment is already complex. For future work, the highest-priority direction is investigating why seed 0 fails systematically. Is it a reward-shaping issue? An initialization problem? An exploration dead-end? A hybrid approach — PD as safety fallback with RL handling the nominal cases — seems like the most pragmatic path forward.

**Suggested figure:** `results/figures/round3_control_energy_mean_std.png` — control energy comparison showing PD's dramatic energy efficiency advantage.

---

## Slide 9: Takeaway Message for Portfolio and Interview

**Title:** The Real Lesson: Scientific Honesty in Applied AI

**Content bullets:**
- **RL is not magic:** a hand-tuned PD controller beats neural network controllers on a real engineering problem
- **Seed variance is the elephant in the room** — most RL papers hide it; this project exposes it
- **The best result is not "RL wins" — it's "here's exactly where RL succeeds and where it fails"**
- **Portfolio value:** demonstrates systematic experimentation, critical analysis, and intellectual honesty
- **Interview soundbite:** "I found that a classical controller beat my RL model in 3 of 4 scenarios. Here's why that's actually the more interesting result."
- **For the AI field:** we need more negative results and more rigorous evaluation protocols if RL is ever going to be trusted in safety-critical domains

**Speaker notes:**
I want to close with what I think is the real value of this project, and it's not the technical results. Any ML engineer can train a PPO agent and report a mean return. The harder thing — and the thing that shows real professional maturity — is to say "my model didn't work as well as a 1980s controller, and here's exactly why." PD dominated because it was designed for this exact control problem. It has no hyperparameters to tune beyond the proportional and derivative gains, no random initialization, no training variance. It just works. PPO has a lot of potential — individual seeds sometimes beat PD — but that potential comes with reliability risk. The seed variance problem I documented is real and it matters. In safety-critical applications, you can't say "it works 67% of the time." What this project demonstrates — and what I'd discuss in an interview — is the discipline to evaluate rigorously, report honestly, and draw the conclusions the data actually supports rather than the ones you hoped for. That's what separates applied AI engineering from academic paper-chasing.

**Suggested figure:** `results/figures/robustness_comparison_round3.png` — side-by-side robustness comparison that visually summarizes the complete results across all scenarios.

---

## Appendix: Quick Reference Table

| Scenario | Winner | Return Delta | Failure Rate Winner |
|---|---|---|---|
| normal_wind | **PD** (−2.59) | PD > PPO mean by 11.9 | PD (0% vs 3.3%) |
| strong_wind | **PD** (−10.10) | PD > PPO mean by 41.9 | PD (0% vs 21.7%) |
| variable_wind | **PD** (−10.90) | PD > PPO mean by 47.8 | PD (0% vs 33.3%) |
| OOD_wind | **PPO+Safety** (−30.19) | PPO wins by 18% | PD (35% vs 40%) |

---

*Presentation outline for OffshoreWind_ControlRL course presentation. Data source: Round 3 results audit (2026-05-13). Figures in `results/figures/` directory.*
