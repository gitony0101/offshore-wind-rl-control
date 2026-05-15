# Round 3 Figure Manifest

**Offshore Wind Turbine Control — RL vs. Classical Controllers**
*Round 3 (Long Training, Multi-Seed) | Generated: 2026-05-13 | Total figures: 8*

This manifest documents every Round 3 figure by filename, what it shows, how to interpret it, and what claim it supports. All descriptions are grounded in the verified findings from `reports/round3_result_audit.md`.

---

## Primary Figures (aggregated evaluation, with error bars)

### 1. `round3_average_return_mean_std.png`

**What it shows:** Grouped bar chart of average episode return (mean +/- 1 std-deviation) for 5 controller groups across 4 wind scenarios. Controllers are PD, PPO, PPO-Randomized, PPO+Safety, and PPO-Randomized+Safety. Scenarios are normal_wind, strong_wind, variable_wind, and out_of_distribution_wind. Higher (less negative) bars are better.

**How to interpret it:** Within each scenario group on the x-axis, compare bar heights across controllers. Error bars show one standard deviation across 3 training seeds. Bars with large error bars indicate high seed variance. Note that PD and NoControl have zero error bars because they are deterministic (no learned parameters).

**Claim supported:** PD dominates in all 3 in-distribution scenarios: normal_wind (-2.59), strong_wind (-10.10), variable_wind (-10.90). PPO+Safety narrowly wins OOD_wind (-30.19 vs PD's -36.78). Large error bars on PPO variants visually demonstrate the seed sensitivity documented in the audit.

---

### 2. `round3_failure_rate_mean_std.png`

**What it shows:** Grouped bar chart of failure rate (mean +/- std) for the same 5 controllers across 4 scenarios. Y-axis is clamped to [0, 1]. Failures are episodes where the platform pitch angle exceeded the 0.3 rad safety threshold.

**How to interpret it:** Bars at or near zero indicate a controller that reliably avoids unsafe platform tilt. Tall bars (0.2-0.4) indicate controllers that fail on 20-40% of episodes. Zero error bar = deterministic behavior (PD) or zero failures; large error bars = seed-dependent failure patterns.

**Claim supported:** PD achieves zero failures in all 3 in-distribution scenarios. NoControl reaches 95-100% failure under stress. All PPO variants exhibit significant failure rates (20-40%) outside normal_wind, and even PPO has 3.3% failure rate in normal_wind. This supports the audit conclusion that RL-based controllers are not safety-reliable without further work.

---

### 3. `round3_control_energy_mean_std.png`

**What it shows:** Grouped bar chart of control energy (mean +/- std) across controllers and scenarios. Y-axis starts at 0. Control energy measures cumulative actuator effort (integral of squared control actions).

**How to interpret it:** Lower bars = less actuator effort = less wear on actuators = lower operational cost. Compare bars within scenarios to see which controller achieves safety with minimal effort. Be cautious: a low-energy controller that also has high failure rate (NoControl) achieves low energy trivially by not controlling at all.

**Claim supported:** PD is dramatically more energy-efficient in normal_wind (25.10 vs. 186+ for PPO variants, roughly 7x less). PD remains the most or second-most efficient in all scenarios. This supports the audit finding that the classical PD controller achieves superior performance with minimal actuator effort, a significant operational advantage.

---

### 4. `round3_mean_abs_pitch_mean_std.png`

**What it shows:** Grouped bar chart of mean absolute pitch angle (|theta|, in radians, mean +/- std) across controllers and scenarios. This measures the average magnitude of platform tilt regardless of direction.

**How to interpret it:** Lower values mean the platform stays closer to level. The safety threshold is 0.3 rad — values approaching this indicate the controller is operating near the failure boundary. Compare across controllers: a lower |theta| while maintaining good return indicates better stabilization.

**Claim supported:** Supports the claim that PD maintains the platform closer to level (smaller average tilt) while using less control energy. PPO variants that use much higher energy but achieve similar or worse pitch angles are clearly suboptimal in the energy-pitch trade-off space.

---

### 5. `round3_robustness_gap.png`

**What it shows:** Bar chart showing the performance degradation from normal_wind to out_of_distribution_wind for each controller. Each bar = OOD avg_return minus normal_wind avg_return. Error bars use propagated uncertainty (quadrature sum of both scenario std values). A horizontal gray dashed line marks zero.

**How to interpret it:** More negative bars = worse degradation when moving from normal to OOD conditions. A bar near zero means the controller handles OOD conditions nearly as well as normal conditions. Bars should be read alongside the failure rate plot — a controller with a small return gap but high failure rate in both scenarios is simply failing in both.

**Claim supported:** Visualizes how much each controller's performance degrades when facing unseen wind conditions. PD shows the expected degradation from its best performance (-2.59) to moderate (-36.78), but PPO variants degrade from similarly poor or worse baselines. Supports the honest conclusion that OOD_wind breaks all controllers to some degree, but PD at least starts from a much stronger in-distribution position.

---

### 6. `round3_learning_curves_by_seed.png`

**What it shows:** Training learning curves (episode reward vs. cumulative timesteps) for 6 models: PPO-Normal and PPO-Randomized, each trained with 3 seeds (0, 1, 2). PPO-Normal seeds use shades of blue (solid for seed 0, dashed for seed 1, dotted for seed 2). PPO-Randomized seeds use shades of orange/gold with the same line style convention.

**How to interpret it:** Each line shows one complete training run (~500K timesteps). Diverging lines within a color family reveal seed variance — lines that end up at very different reward levels indicate that random initialization critically affects final policy quality. Lines that stay flat or go deeply negative indicate failed training runs.

**Claim supported:** This is the most important figure for the seed variance finding. It visually confirms the audit's "seed 0 problem" — seed 0 lines for both PPO-Normal and PPO-Randomized consistently underperform seeds 1 and 2. If any lines plateau at very different levels, it demonstrates the bimodal distribution that makes PPO unreliable for safety-critical deployment.

---

## Legacy Round 3 Figures (raw per-episode data, no error bars)

### 7. `robustness_comparison_round3.png`

**What it shows:** Grouped bar chart of avg_return by controller and scenario, generated by pivoting the raw `evaluation_summary_round3.csv` (57 rows: 5 controllers + NoControl across 4 scenarios, 20 episodes each, 3 seeds for PPO variants). Unlike figure 1, this plot does not aggregate across seeds — it shows raw per-run data without error bars.

**How to interpret it:** Similar to figure 1 but without error bars. May show slightly different aggregation behavior since it pivots raw data rather than pre-aggregated means. Use this as a secondary view to verify the patterns seen in the error-bar version.

**Claim supported:** Reinforces the same claims as figure 1 (PD dominance in-distribution, PPO seed variance) but presented without standard deviation context. Useful for readers who want to see raw pivot data.

---

### 8. `failure_rate_bar_round3.png`

**What it shows:** Bar chart of failure rate by controller and scenario from raw `evaluation_summary_round3.csv`. Similar to figure 2 but without error bars — it shows the failure rate for each controller-scenario combination without aggregating seeds.

**How to interpret it:** Same axes as figure 2 but without std error bars. May have different bar heights if the raw pivot treats the data differently from the aggregated version.

**Claim supported:** Reinforces the same claims as figure 2 (PD zero failures in-distribution, PPO variants 20-40% failure) but without statistical context. The lack of error bars here is a weakness compared to the `round3_failure_rate_mean_std.png` version.

---

## Summary Table

| # | Filename | Chart Type | Key Claim |
|---|----------|-----------|-----------|
| 1 | `round3_average_return_mean_std.png` | Grouped bar + error bars | PD wins 3/4 scenarios; PPO+Safety narrows OOD |
| 2 | `round3_failure_rate_mean_std.png` | Grouped bar + error bars | PD zero failures in-distribution; PPO unreliable |
| 3 | `round3_control_energy_mean_std.png` | Grouped bar + error bars | PD 7x more energy-efficient in normal_wind |
| 4 | `round3_mean_abs_pitch_mean_std.png` | Grouped bar + error bars | PD stabilizes better with less effort |
| 5 | `round3_robustness_gap.png` | Bar + error bars | OOD degrades all controllers; PD starts stronger |
| 6 | `round3_learning_curves_by_seed.png` | Multi-line time series | Seed 0 systematically underperforms |
| 7 | `robustness_comparison_round3.png` | Grouped bar (raw) | Secondary view of return comparison |
| 8 | `failure_rate_bar_round3.png` | Grouped bar (raw) | Secondary view of failure rates |

---

*Manifest compiled by Hermes Agent (Nous Research) | 2026-05-13 | Source: `reports/round3_result_audit.md` and `src/analysis/plot_results.py`*
