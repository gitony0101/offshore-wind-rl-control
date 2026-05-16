# Methodology: Reinforcement Learning for Floating Offshore Wind Platform Stabilization

## 1. Business and Technical Problem

Floating offshore wind turbines represent a rapidly growing segment of renewable energy. Unlike bottom-fixed turbines, floating platforms rest on semi-submersible, spar, or tension-leg-pile moorings, allowing deployment in deeper waters with stronger, more consistent winds.

However, floating platforms are subject to significant pitch motion driven by:

- **Aerodynamic forces**: wind shear, gusts, and turbulence acting on the rotor and tower
- **Hydrodynamic forces**: wave loading, radiation damping, and mooring-line compliance

Excessive pitch angles lead to structural fatigue, reduced aerodynamic efficiency, power quality degradation, and in extreme cases, catastrophic failure. The control challenge is to stabilise the platform while respecting structural limits and maintaining energy production.

## 2. Evidence Boundary

### Inspiration Sources

This project draws conceptual inspiration from several publicly available research efforts:

| Source | Concept Used | Original Implementation |
|--------|-------------|------------------------|
| **DeepSense** | General concept of deep learning for ocean/sensor data | Conceptual inspiration only; all code is original |
| **RL-PSF** | Idea of reinforcement learning with safety/predictive safety filtering | Safety filter logic is independently designed |
| **FloatingFarmYaw** | Concept of RL for floating wind farm control (yaw) | Environment, dynamics, and RL formulation are original |

### What Is Original

All implementation details in this project are **independently and originally written**, including:

- The Gymnasium environment (`FloatingPlatformEnv`)
- The dynamics model (simplified mass-spring-damper)
- The reward function design and weight tuning
- The safety filter algorithm
- The baseline controllers (NoControl, PD)
- The PPO training and evaluation pipeline
- The evaluation metrics and analysis scripts

### What Is Inspired

- The **general problem framing** of applying RL to floating wind control comes from the broader research literature.
- The **concept** of a predictive safety filter comes from the RL-PSF family of work.
- The **idea** of using RL for floating wind stability (rather than yaw optimisation) is informed by projects like FloatingFarmYaw.

> **Important**: No code was copied from any external repository. All implementation choices are independently documented in this project.

## 3. Key Simplifications

This project intentionally uses a **highly simplified** representation of the floating wind control problem:

| Simplification | Rationale |
|----------------|-----------|
| **Single-axis pitch only** | Real platforms exhibit 6 degrees of freedom (surge, sway, heave, roll, pitch, yaw). We model only pitch. |
| **Abstract mass-spring-damper dynamics** | The dynamics use a linear second-order ODE instead of the coupled aero-hydro-servo-elastic equations solved by OpenFAST. |
| **No aerodynamic model** | No blade-element momentum theory, no wake effects, no power-curve mapping. Wind is represented as a scalar disturbance proxy. |
| **No wave spectrum model** | Ocean waves are represented as Gaussian noise rather than a JONSWAP or Pierson-Moskowitz spectrum. |
| **Single control input** | The action is a single continuous scalar, not blade-pitch angles, generator torque, or individual actuator commands. |
| **No turbine-level or farm-level simulation** | No power output model, no structural fatigue model, no multi-turbine interactions. |

These simplifications make the environment fast, stable, and easy to reason about, but they also mean the results are **qualitative demonstrations** rather than engineering-grade predictions.

## 4. Reinforcement Learning Formulation

### 4.1 Markov Decision Process (MDP)

The problem is formalised as a discrete-time Markov Decision Process defined by the tuple $(S, A, R, T, \gamma)$:

- **$S$**: State space
- **$A$**: Action space
- **$R$**: Reward function
- **$T$**: Transition dynamics
- **$\gamma = 0.99$**: Discount factor

### 4.2 State Space

At each timestep, the observation is a 4-dimensional vector:

$$s_t = [\theta_t, \dot{\theta}_t, w_t^{\text{wind}}, w_t^{\text{wave}}]$$

| Component | Symbol | Units | Description |
|-----------|--------|-------|-------------|
| Pitch angle | $\theta$ | rad | Platform pitch deviation from upright |
| Angular velocity | $\dot{\theta}$ | rad/s | Rate of pitch rotation |
| Wind disturbance | $w^{\text{wind}}$ | N·m (proxy) | Stochastic wind force proxy at current step |
| Wave disturbance | $w^{\text{wave}}$ | N·m (proxy) | Stochastic wave force proxy at current step |

The observation space is unbounded (`Box(-inf, inf, shape=(4,), dtype=float32)`), since disturbances are drawn from unbounded Gaussian distributions.

### 4.3 Action Space

The action is a single continuous value normalised to $[-1, 1]$:

$$a_t \in [-1, 1]$$

The action maps to a physical control force via a gain parameter:

$$f^{\text{control}}_t = a_t \times \text{action\_gain}$$

where `action_gain = 0.5` by default.

### 4.4 Transition Dynamics

The platform dynamics follow a simplified discrete-time second-order model:

$$\theta_{t+1} = \theta_t + \dot{\theta}_t \cdot \Delta t$$

$$\dot{\theta}_{t+1} = \dot{\theta}_t + \frac{\Delta t}{m} \left(w_t^{\text{wind}} + w_t^{\text{wave}} + f^{\text{control}}_t - c \cdot \dot{\theta}_t - k \cdot \theta_t\right)$$

where:
- $m$ = generalized mass (default: 1.0 kg)
- $c$ = viscous damping coefficient (default: 0.5 N·m·s/rad)
- $k$ = restoring stiffness coefficient (default: 1.5 N·m/rad)
- $\Delta t$ = timestep (default: 0.05 s, i.e., 20 Hz)

Integration is performed via forward Euler.

### 4.5 Reward Function

The reward penalises undesirable behaviour at each step:

$$r_t = -\theta_{t+1}^2 - 0.5\dot{\theta}_{t+1}^2 - 0.1 f_{\text{control}, t}^2 + r_{\text{safety}}$$

where the safety bonus/penalty is:

$$
r_{\text{safety}} = \begin{cases}
-10.0 & \text{if } |\theta_{t+1}| > 0.9 \times \theta_{\text{safety}} \\
0 & \text{otherwise}
\end{cases}
$$

with $\theta_{\text{safety}} = 0.3$ rad (~17 degrees).

| Component | Weight | Rationale |
|-----------|--------|-----------|
| Position penalty | $-1.0 \times \theta^2$ | Primary objective: keep platform upright |
| Velocity penalty | $-0.5 \times \dot{\theta}^2$ | Penalise rapid oscillations |
| Control energy penalty | $-0.1 \times f_{\text{control}}^2$ | Discourage excessive actuator use |
| Safety pre-warning | $-10.0$ (one-shot) | Strong negative signal near the boundary |

### 4.6 Done Conditions

An episode terminates when:

1. **Safety violation**: $|\theta| > \theta_{\text{safety}}$ (hard termination with `terminated=True`)
2. **Time horizon**: episode reaches `max_steps` (default: 1000 steps = 50 seconds) → `truncated=True`

### 4.7 Episode Initial Conditions

At each reset:
- $\theta \sim \mathcal{U}(-0.1, 0.1)$ (~±5.7 degrees)
- $\dot{\theta} \sim \mathcal{U}(-0.05, 0.05)$ rad/s
- Wind and wave disturbances are immediately sampled to populate the observation vector

## 5. Wind and Wave Model

Disturbances are modelled as **stochastic Gaussian noise** with configurable statistics per scenario:

$$w^{\text{wind}}_t \sim \mathcal{N}(\mu_{\text{wind}}, \sigma_{\text{wind}})$$
$$w^{\text{wave}}_t \sim \mathcal{N}(0, \sigma_{\text{wave}})$$

### Scenario Configuration

| Scenario | wind_std | wave_std | Description |
|----------|----------|----------|-------------|
| `normal_wind` | 0.3 | 0.2 | Baseline operating conditions |
| `strong_wind` | 0.6 | 0.4 | Elevated wind and wave activity |
| `variable_wind` | 0.3 | 0.2 | Wind mean ramps from 0 → 0.5 over the episode |
| `out_of_distribution_wind` | 0.9 | 0.7 | Extreme disturbance levels for robustness testing |

### Domain Randomisation

When `randomized_training=True`, the following parameters are resampled at each reset:

| Parameter | Range |
|-----------|-------|
| `stiffness` | [0.8, 2.2] |
| `damping` | [0.2, 0.8] |
| `mass` | [0.5, 1.5] |
| `wind_std` | [0.1, 2× default] |
| `wave_std` | [0.05, 2× default] |

## 6. Baselines

### 6.1 NoControl

The simplest possible baseline. The controller always outputs zero action:

$$a_t = 0$$

**Purpose**: Reveal the uncontrolled dynamics of the platform. Under sufficient disturbance, the platform should exhibit significant pitch motion and potentially fail.

### 6.2 PD Controller

A classical Proportional-Derivative controller:

$$a_t = \text{clip}\left(-K_p \cdot \theta_t - K_d \cdot \dot{\theta}_t, -1, 1\right)$$

Default gains: $K_p = 5.0$, $K_d = 2.0$.

**Purpose**: Provide a non-learning reference. PD control is analytically well-understood for second-order systems, and any competent RL policy should achieve comparable or better performance.

## 7. PPO Method

### Algorithm

**Proximal Policy Optimization** (PPO) as implemented in Stable-Baselines3.

### Policy Architecture

- **Policy type**: `MlpPolicy` (multi-layer perceptron)
- **Network**: Default SB3 architecture — two hidden layers of 64 units each with tanh activation
- **Actor output**: Mean and standard deviation of Gaussian action distribution
- **Critic output**: Single value estimate

### Default Hyperparameters

| Parameter | Value |
|-----------|-------|
| `learning_rate` | 3e-4 |
| `n_steps` | 2048 |
| `batch_size` | 64 |
| `n_epochs` | 10 |
| `gamma` | 0.99 |
| `gae_lambda` | 0.95 |
| `clip_range` | 0.2 |
| `ent_coef` | 0.0 |
| `vf_coef` | 0.5 |
| `max_grad_norm` | 0.5 |

### Training Protocol

1. Create a vectorized environment (`DummyVecEnv`) wrapping `FloatingPlatformEnv`
2. Instantiate PPO with default hyperparameters
3. Train for a specified number of timesteps (default: 50,000; range: 1,000 smoke-test to 1,000,000 production)
4. Save model checkpoint every 25% of total timesteps
5. Save final model to `results/models/ppo_{scenario}.zip`
6. Extract training log from episode info buffer

## 8. Evaluation

### 8.1 Scenarios

All controllers are evaluated across four scenarios:

| Scenario | Purpose |
|----------|---------|
| `normal_wind` | Baseline performance under nominal conditions |
| `strong_wind` | Performance under elevated disturbances |
| `variable_wind` | Adaptability to changing wind conditions |
| `out_of_distribution_wind` | Robustness to unseen extreme conditions |

### 8.2 Controllers Evaluated

| Controller | Type | Description |
|------------|------|-------------|
| NoControl | Baseline | Zero action |
| PD | Baseline | Classical PD feedback (Kp=5.0, Kd=2.0) |
| PPO | RL | Trained Stable-Baselines3 PPO agent |
| PPO+Safety | RL + Filter | PPO wrapped in simple safety filter |

### 8.3 Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `avg_return` | $\frac{1}{N}\sum_{i=1}^N R_i$ | Average cumulative reward per episode |
| `mean_abs_theta` | $\mathbb{E}[|\theta|]$ | Average pitch deviation over all steps |
| `max_abs_theta` | $\max|\theta|$ | Worst-case pitch angle observed |
| `failure_rate` | $\frac{\text{# safety violations}}{N_{\text{episodes}}}$ | Fraction of episodes ending in safety violation |
| `control_energy` | $\mathbb{E}\left[\sum_t a_t^2\right]$ | Mean actuation effort per episode |

### 8.4 Evaluation Protocol

- **Episodes per scenario**: 20 (configurable via `--episodes`)
- **Random seed**: 42 (configurable)
- **Output**: CSV summary saved to `results/metrics/evaluation_summary.csv`
- **Trajectories**: Full step-by-step data for the first episode saved per scenario

## 9. Safety Filter

### Concept

The safety filter implements a **one-step-ahead model predictive safety check**:

1. Given the current state and the proposed action, integrate the dynamics model forward by one timestep (Euler method)
2. If the predicted pitch angle exceeds the safety threshold ($|\theta_{\text{pred}}| > 0.3$): **emergency stop** — action forced to zero
3. If the predicted pitch angle enters a pre-warning zone ($|\theta_{\text{pred}}| > 0.24$, i.e. 80% of threshold): **scale down** — action reduced by 50% and clamped
4. Otherwise: **pass through** — action unchanged

### Implementation

- `safety_filter()` function: standalone one-step prediction and intervention logic
- `SafetyFilteredController` class: wrapper around any controller with a `predict()` method
- Tracks intervention count for analysis

### Limitations

- Single-step lookahead (not multi-step MPC)
- Uses the simplified dynamics model (potential model mismatch)
- Does not account for future disturbance evolution
- Graded response (emergency stop vs. scale-down) is heuristic

## 10. Round 2 Analysis

### 10.1 Why PD Outperformed Short-Trained PPO

Round 1 evaluation showed the PD controller (Kp=5.0, Kd=2.0) achieving 0% failure rate across normal, strong, and variable wind scenarios, while PPO trained for only 5,000 timesteps was inconsistent. This section explains why this result is expected and what it tells us about classical vs. learning-based control.

#### The Short-Training Gap

PPO is a policy gradient method that starts from randomly initialised neural network weights. At each training timestep, it:
1. Rolls out trajectories using the current policy
2. Computes advantage estimates (Generalised Advantage Estimation)
3. Updates the policy network via gradient descent with a clipped surrogate objective

At 5,000 timesteps with `n_steps=2048`, only about 2-3 gradient update cycles have occurred. The policy has seen roughly 40-60 episodes of experience. This is far too little for the network to converge to an effective control strategy.

In contrast, the PD controller is computed directly from the analytical structure of a second-order system:

$$ a_t = -K_p \cdot \theta_t - K_d \cdot \dot{\theta}_t $$

This is not "learning" -- it is a closed-form control law with guaranteed stability for the linear system when gains are chosen appropriately. The gains Kp=5.0, Kd=2.0 were selected to produce an overdamped response with adequate disturbance rejection.

#### Implications

This result does not indicate that RL is inferior to classical control. It highlights:

| Aspect | PD Control | PPO (short-training) | PPO (extended training) |
|--------|-----------|---------------------|------------------------|
| Competence at low training | Immediate | Low | Improves with timesteps |
| Design effort | Requires analytical expertise | None (but requires hyperparameter tuning) | None |
| Scalability to high-dimensional | Difficult | Good | Good |
| Multi-objective optimisation | Requires manual weight tuning | Learns from reward | Learns from reward |
| Robustness to parametric uncertainty | Fixed (may degrade) | Brittle without randomisation | Improved with domain randomisation |

Round 2 addresses the short-training gap by extending PPO training to 50,000 timesteps (10x the previous budget).

### 10.2 Domain Randomisation: Theory and Practice

**Definition**: Domain randomisation is a training technique where environment parameters are randomly varied at each episode reset, exposing the agent to a distribution of related environments rather than a single fixed environment.

**Mathematical formulation**: Instead of optimising the policy in a single MDP $M$, the agent is trained on a distribution of MDPs $\{M^{(i)}\}_{i=1}^{N}$ where each $M^{(i)}$ has different transition dynamics $T^{(i)}$ parameterised by randomised physical parameters:

$$ \pi^* = \arg\max_\pi \mathbb{E}_{M \sim p(M)} \left[ \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t r_t \mid M \right] \right] $$

**Parameters randomised in this project**:

| Parameter | Nominal Value | Randomisation Distribution |
|-----------|--------------|---------------------------|
| Stiffness $k$ | 1.5 N.m/rad | $\mathcal{U}(0.8, 2.2)$ |
| Damping $c$ | 0.5 N.m.s/rad | $\mathcal{U}(0.2, 0.8)$ |
| Mass $m$ | 1.0 kg | $\mathcal{U}(0.5, 1.5)$ |
| Wind std | scenario default | $\mathcal{U}(0.1, 2 \times \sigma_{\text{default}})$ |
| Wave std | scenario default | $\mathcal{U}(0.05, 2 \times \sigma_{\text{default}})$ |

**Why domain randomisation matters**:

1. **Robustness**: A policy trained on a single parameter set may exploit quirks of that specific configuration. Randomisation forces the policy to learn behaviour that works across a family of systems.

2. **Implicit regularisation**: Randomised training acts as a form of regularisation, preventing the policy from overfitting to narrow state-action trajectories. The resulting policy tends to be smoother and more conservative.

3. **Sim-to-real transfer**: Simulated environments never perfectly match reality. Domain randomisation teaches the policy to handle parametric uncertainty, which is a prerequisite for deploying learned policies on real hardware. The OpenAI Dactyl project (OpenAI et al., 2019) demonstrated this by training a robotic hand policy entirely in simulation with domain randomisation and deploying it on a physical system.

4. **Wider training distribution**: Each episode presents a slightly different dynamical system, increasing the diversity of state transitions the policy encounters. This expands the effective support of the training data.

**What domain randomisation does NOT do**:
- It does not guarantee robustness to all parameter combinations outside the randomisation range
- It does not replace the need for adequate training time
- It does not address structural model errors (e.g., missing physics modes)
- It is not a substitute for formal safety guarantees

### 10.3 Safety Filter Design and Limitations

#### Design

The safety filter implements a one-step-ahead model predictive safety check:

$$ \theta_{\text{pred}} = \theta_t + \dot{\theta}_t \cdot \Delta t + \frac{\Delta t^2}{m} \left( f_{\text{control}} + w_t^{\text{wind}} + w_t^{\text{wave}} - c \cdot \dot{\theta}_t - k \cdot \theta_t \right) $$

Intervention logic:

| Condition | Intervention |
|-----------|-------------|
| $|\theta_{\text{pred}}| > 0.3$ rad | Emergency stop: $a = 0$ |
| $|\theta_{\text{pred}}| > 0.24$ rad (80% of threshold) | Scale action by 0.5, clamp to [-1, 1] |
| Otherwise | Pass through unchanged |

#### Known Limitations

1. **Single-step horizon**: The filter only predicts one timestep ahead. A sequence of marginally safe actions could compound into a violation across multiple steps. Multi-horizon MPC would address this.

2. **Model identity**: The prediction uses the same dynamics model as the environment. In a real deployment, the safety model should be more conservative and higher-fidelity than the plant model to account for model error.

3. **Constant disturbance assumption**: The prediction assumes disturbances remain constant over the lookahead horizon. Sudden gusts or wave impacts are not anticipated.

4. **Heuristic thresholds**: The 80% pre-warning zone and 50% scaling factor are heuristic choices, not derived from formal analysis.

#### Round 2 Evaluation Approach

Round 2 does not modify the safety filter's core logic. Instead, it evaluates the filter more rigorously:
- Track per-episode intervention counts (emergency stops + action scalings)
- Compare PPO+Safety failure rates against raw PPO across all four scenarios
- Assess the performance trade-off (return reduction vs. safety improvement)

#### Recommended Future Work

| Approach | Description | Formal Guarantees? |
|----------|-------------|-------------------|
| Multi-horizon MPC safety | Receding-horizon optimisation with constraint satisfaction | Yes (over the prediction horizon) |
| Control Barrier Functions (CBFs) | Lyapunov-based safety filter with formal forward-invariance guarantees | Yes |
| Constrained RL (CPO, PPO-Lag) | Safety constraints baked into the training objective | Partially (in expectation) |
| Hamilton-Jacobi reachability | Computed safe set via Hamilton-Jacobi-Bellman equations | Yes (for low-dimensional systems) |

---

## 11. Round 3 Analysis: Multi-Seed Evaluation and Extended Training

### 11.1 Why Multi-Seed Evaluation Is Necessary

All previous rounds of evaluation used a single random seed (seed 42). This is problematic for three reasons:

1. **Policy gradient variance**: PPO's trajectory sampling, advantage estimation, and gradient updates are all stochastic. Two training runs with different initialisation seeds can produce meaningfully different policies even with identical hyperparameters and training budgets.

2. **Seed-dependent evaluation noise**: Episode-level disturbances are drawn from a seeded RNG. A single test seed can produce a lucky or unlucky disturbance sequence, inflating or deflating observed failure rates purely by chance.

3. **Statistical reliability**: Without reporting means and standard deviations across multiple independent seeds, it is impossible to distinguish genuine performance differences from random variation. This is a well-documented problem in the RL literature (Henderson et al., 2018).

Best practice is to evaluate across at least 3-5 seeds and report mean ± standard deviation. Round 3 adopts this standard.

### 11.2 Why Longer PPO Training (500k Timesteps) Is Needed

The training budget progression from Round 1 to Round 3 reflects a deliberate investigation into the **training-time to performance** curve:

| Training Budget | Approx. Gradient Updates | Expected Policy Maturity |
|-----------------|-------------------------|--------------------------|
| 5,000 (Round 1) | ~2-3 | Undertrained, still exploring |
| 50,000 (Round 2) | ~25 | Converging, possibly plateauing |
| 500,000 (Round 3) | ~250 | Mature, should approach asymptotic performance |

For low-dimensional control tasks where a well-tuned PD controller provides an analytically near-optimal solution, PPO needs sufficient experience to (a) explore enough of the state space to discover the stabilising control law, and (b) refine the neural network weights to match the precision of the closed-form controller.

The 500,000 timestep budget tests whether PPO can eventually close the performance gap with PD given adequate training.

### 11.3 Interpretation Framework

The following questions structure the Round 3 analysis:

1. **Does PPO close the gap with PD at 500k timesteps?** If PD still wins across most metrics, this is a valid finding — classical control demonstrably remains superior for low-dimensional linear systems. This does not represent a failure of RL; rather, it confirms the "no free lunch" principle and clarifies where model-free methods are and are not competitive.

2. **Does domain randomisation improve OOD robustness?** The key metric is the **robustness gap**: the performance degradation when moving from in-distribution (normal_wind) evaluation to out-of-distribution (strong_wind, ood_wind) evaluation. If domain-randomised PPO shows a smaller robustness gap compared to standard PPO, domain randomisation has achieved its intended effect.

3. **Do the results support or weaken the case for model-free RL?** The honesty requirement here is critical. If PPO at 500k still underperforms PD even with domain randomisation and multi-seed averaging, the honest conclusion is that model-free RL offers no practical advantage for this specific task formulation. That is a scientifically valid outcome. The case for RL in wind platform control may depend on factors not captured by this simplified environment (nonlinear aero-hydro coupling, multi-objective optimisation, high-dimensional sensor inputs).

### 11.4 Verified Results

A multi-seed evaluation was conducted: 3 seeds (0, 1, 2), 500,000 training timesteps per seed, 4 wind scenarios, 20 evaluation episodes. Six controller configurations were tested across 1,680 total evaluation episodes. The following results are verified from `evaluation_summary_round3.csv` (57 rows) and `evaluation_summary_round3_aggregated.csv` (24 rows).

#### 11.4.1 Per-Scenario Rankings (by avg_return)

| Rank | normal_wind | strong_wind | variable_wind | OOD_wind |
|------|-------------|-------------|--------------|----------|
| **1** | **PD (-2.59)** | **PD (-10.10)** | **PD (-10.90)** | **PPO+Safety (-30.19)** |
| 2 | NoControl (-9.40) | PPO-Rand+Safety (-31.26) | PPO (-58.73) | PD (-36.78) |
| 3 | PPO-Randomized (-12.05) | PPO-Randomized (-37.13) | PPO-Rand+Safety (-62.04) | PPO-Randomized (-36.87) |
| 4 | PPO (-14.49) | PPO+Safety (-43.82) | PPO-Randomized (-71.59) | PPO-Rand+Safety (-38.41) |
| 5 | PPO-Rand+Safety (-16.29) | PPO (-51.98) | PPO+Safety (-74.32) | PPO (-42.01) |
| 6 | PPO+Safety (-19.83) | NoControl (-97.73) | NoControl (-201.86) | NoControl (-49.77) |

**Winner declarations with evidence quality:**

| Scenario | Winner | Evidence Quality |
|----------|--------|-----------------|
| `normal_wind` | **PD** (-2.59) | Very strong -- PD beats all PPO variants by >4.6x return margin, with zero failures and 7x less control energy |
| `strong_wind` | **PD** (-10.10) | Strong -- PD beats best PPO variant (PPO-Rand+Safety at -31.26) by >3x margin, zero failures vs 20-30% |
| `variable_wind` | **PD** (-10.90) | Very strong -- PD beats best PPO (PPO at -58.73) by >5x margin, zero failures vs 33% |
| `OOD_wind` | **PPO+Safety** (-30.19) | Weak/narrow -- beats PD by 18%, but at higher failure rate (40% vs 35%) with only n=3 seeds |

#### 11.4.2 PD Dominance: In-Distribution Performance

PD achieves the following across the three in-distribution scenarios:

| Metric | normal_wind | strong_wind | variable_wind |
|--------|-------------|-------------|---------------|
| avg_return | -2.59 | -10.10 | -10.90 |
| std | 0.0 | 0.0 | 0.0 |
| failure_rate | 0.0% | 0.0% | 0.0% |
| control_energy | 25.10 | 98.80 | 154.35 |

Key observations:
- PD achieves **zero variance** and **zero failures** in all in-distribution scenarios -- it is a deterministic, analytically-derived controller with no learned parameters to introduce variability
- PD uses **7x less control energy** than PPO variants in normal_wind (25.10 vs 186+), and ~5x less in strong/variable wind. This translates to lower actuator wear
- The best PPO variant in normal_wind averages -12.05 (5x worse return than PD)

#### 11.4.3 PPO Seed Variance: The Critical Finding

PPO policies exhibit extreme seed sensitivity. The following table shows the range of average returns across 3 seeds for each PPO variant:

| Variant | normal_wind range | strong_wind range | variable_wind range | OOD_wind range |
|---------|-------------------|-------------------|--------------------|----------------|
| PPO | -39.11 to -2.17 (range=36.95) | -139.88 to -7.98 (range=131.90) | -149.85 to -11.62 (range=138.23) | -82.72 to -21.50 (range=61.22) |
| PPO+Safety | -55.11 to -2.18 (range=52.93) | -115.38 to -8.00 (range=107.38) | -196.60 to -11.65 (range=184.95) | -47.23 to -21.30 (range=25.93) |
| PPO-Randomized | -31.64 to -2.19 (range=29.45) | -95.65 to -7.80 (range=87.85) | -188.77 to -12.35 (range=176.42) | -66.79 to -21.75 (range=45.04) |
| PPO-Rand+Safety | -44.33 to -2.24 (range=42.09) | -78.02 to -7.81 (range=70.21) | -160.15 to -12.34 (range=147.81) | -72.53 to -21.21 (range=51.32) |

**The Seed 0 Problem:**

Across all scenarios and all PPO variants, **seed 0 consistently fails catastrophically:**

| Seed 0 Performance | normal_wind | strong_wind | variable_wind | OOD_wind |
|--------------------|-------------|-------------|--------------|----------|
| PPO | -39.11 (10% fail) | -139.88 (65% fail) | 100% fail | 100% fail |
| PPO-Randomized | -31.64 (5% fail) | -95.65 (90% fail) | 100% fail | 100% fail |
| PPO+Safety | -55.11 (5% fail) | -115.38 (85% fail) | 100% fail | 100% fail |
| PPO-Rand+Safety | -44.33 (0% fail) | -78.02 (80% fail) | 100% fail | 100% fail |

In contrast, **seeds 1 and 2 often achieve returns competitive with PD** (e.g., PPO-Randomized-seed1 in strong_wind reaches -7.80, beating PD's -10.10).

**Interpretation:** The average return across 3 seeds is misleading because it conceals a bimodal distribution: seeds 1-2 often perform well, while seed 0 consistently fails. This is a structural pattern, not a fluke -- reporting a single seed (especially seed 1 or 2) would cherry-pick a favorable outcome, while reporting the mean would obscure the bimodality.

#### 11.4.4 Failure Rates Across All Scenarios

| Controller | normal_wind | strong_wind | variable_wind | OOD_wind | avg_fail |
|-----------|-------------|-------------|--------------|----------|----------|
| PD | 0.0% | 0.0% | 0.0% | 35.0% | **8.8%** |
| PPO | 3.3% | 21.7% | 33.3% | 40.0% | 24.6% |
| PPO+Safety | 1.7% | 28.3% | 33.3% | 40.0% | 25.8% |
| PPO-Randomized | 1.7% | 30.0% | 33.3% | 40.0% | 26.3% |
| PPO-Rand+Safety | 0.0% | 26.7% | 33.3% | 40.0% | 25.0% |
| NoControl | 0.0% | 95.0% | 100.0% | 100.0% | 73.8% |

PD is the only controller that achieves zero failures across all three in-distribution scenarios. All PPO variants cluster at 20-40% failure rates outside normal_wind.

#### 11.4.5 Safety-Filtered PPO in OOD Wind

OOD_wind is the only scenario where a PPO variant wins:

| Controller | avg_return | std | failure_rate | control_energy |
|------------|-----------|-----|-------------|---------------|
| **PPO+Safety** | **-30.19** | 12.05 | 40.0% | 164.56 |
| PD | -36.78 | 0.00 | 35.0% | 189.58 |
| PPO-Randomized | -36.87 | 21.16 | 40.0% | 198.58 |
| PPO-Rand+Safety | -38.41 | 24.12 | 40.0% | 201.62 |
| PPO | -42.01 | 28.79 | 40.0% | 171.36 |
| NoControl | -49.77 | 0.00 | 100.0% | 0.00 |

**Caveats on the OOD win:**

1. The margin over PD is 18% in return (-30.19 vs -36.78), but PPO+Safety has a **higher** failure rate (40% vs 35%)
2. The result is based on only n=3 seeds with no statistical significance testing
3. PPO+Safety had an intervention rate of 42% in OOD, meaning the safety layer was active on a substantial fraction of steps -- it is unclear how much of the return is from the policy vs. the filter
4. OOD_wind breaks all controllers: even the winner fails in 40% of episodes

#### 11.4.6 Domain Randomisation: Mixed Results

| Scenario | PPO avg | PPO-Randomized avg | Randomized better? |
|----------|---------|--------------------|--------------------|
| normal_wind | -14.49 | -12.05 | Yes (slightly) |
| strong_wind | -51.98 | -37.13 | Yes |
| variable_wind | -58.73 | -71.59 | **No (worse)** |
| OOD_wind | -42.01 | -36.87 | Yes (slightly) |

Domain randomisation helps in strong_wind and OOD_wind but **worsens** performance in variable_wind. It does **not** solve the seed-0 failure problem.

#### 11.4.7 Interpretation

The verified results answer the three questions from Section 11.3:

1. **Does PPO close the gap with PD at 500k timesteps?** No. PD remains the highest-performing controller on average return across all four scenarios. Even the best individual PPO seeds (1 and 2) only match PD in some scenarios, and seed 0's catastrophic failures drag the mean far below PD's deterministic performance.

2. **Does domain randomisation improve OOD robustness?** Partially and inconsistently. PPO-Randomized improves on standard PPO in strong_wind (39% better return) and OOD_wind (12% better), but worsens performance in variable_wind (22% worse return). The seed-0 failure pattern persists across both standard and randomized variants.

3. **Do the results support or weaken the case for model-free RL?** These results weaken the case for model-free RL on this specific task. On a simplified, low-dimensional, approximately linear second-order system, classical PD control outperforms PPO at 500k timesteps by every practical measure: higher return, lower failure rate, zero variance, and dramatically lower energy consumption. This is a scientifically valid "no free lunch" finding -- classical control is analytically near-optimal for linear systems, and RL's value proposition emerges in high-dimensional, nonlinear, or multi-objective settings where analytical methods become intractable.

---

## 12. Limitations

This methodology has several important limitations:

1. **Simplified physics**: The mass-spring-damper model does not capture the full aero-hydro-servo-elastic coupling of a real floating wind turbine. Real systems exhibit nonlinear restoring forces, wave-frequency and slow-drift motions, and coupling between degrees of freedom.

2. **Abstract disturbance model**: Wind and wave are modelled as Gaussian noise. Real ocean environments have structured spectral content (JONSWAP, Pierson-Moskowitz), directional effects, and spatial correlations.

3. **No power/fatigue model**: The reward function does not include energy production or fatigue damage accumulation, which are central objectives in real wind turbine control.

4. **Single turbine, single axis**: No inter-turbine wake interactions, no multi-DOF dynamics, no platform-heave or tower-bending modes.

5. **Small-scale training**: Default training is 50,000 timesteps — sufficient for convergence in this simplified environment but not representative of production-scale RL training.

6. **No transfer-to-reality**: The sim-to-real gap is not addressed. The abstract model is too far from reality for direct deployment.

## 13. Future Work

The following extensions are recommended for future iterations:

| Extension | Description | Priority |
|-----------|-------------|----------|
| **SAC / TD3** | Off-policy algorithms may improve sample efficiency and handle continuous control more robustly | Medium |
| **OpenFAST integration** | Replace the abstract dynamics with a full NREL OpenFAST simulation for engineering-fidelity testing | High |
| **Multi-turbine / farm-level** | Extend to multiple turbines with wake interactions using FLORIS or FAST.Farm | Low |
| **Real-world data** | Validate against measured platform telemetry from operational floating wind farms | Low |
| **MPC-based safety filter** | Replace the heuristic safety filter with a proper Model Predictive Control safety layer | Medium |
| **Domain randomisation → sim2real** | Systematically study how much randomisation is needed for policies to transfer to higher-fidelity environments | Medium |
| **Multi-objective reward** | Include power output maximisation and fatigue minimisation in the reward | High |
| **Constrained RL** | Use Constrained Policy Optimization (CPO) or Lagrangian methods for formal safety guarantees | Medium |
| **Robustness analysis** | Formal robustness testing with structured uncertainty and adversarial disturbance patterns | Low |

---

## Disclaimer

> **This is a simplified simulation based prototype for learning and portfolio purposes. It does not claim engineering fidelity to real floating offshore wind turbine systems.**
>
> The results presented in this project are qualitative demonstrations of RL concepts applied to a simplified floating offshore wind platform model. They should not be interpreted as predictions of real-world turbine behaviour, nor should they be used for engineering design, safety certification, or operational decision-making.
