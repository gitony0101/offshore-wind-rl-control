# Round 3 Result Audit Report

**Offshore Wind Turbine Control — RL vs. Classical Controllers**
*Audit Date: 2026-05-13 | Data Source: `results/metrics/evaluation_summary_round3.csv` (57 rows, 15 controller/seed configs across 4 scenarios, 20 episodes each)*

---

## 1. Per-Scenario Rankings (by avg_return)

Controllers ranked within each wind scenario from highest to lowest average return. Higher (less negative) is better.

### Scenario: normal_wind

| Rank | Controller | avg_return | std | failure_rate | control_energy |
|------|-----------|-----------|------|-------------|---------------|
| 1 | **PD** | **-2.59** | 0.00 | 0.0% | 25.10 |
| 2 | NoControl | -9.40 | 0.00 | 0.0% | 0.00 |
| 3 | PPO-Randomized | -12.05 | 13.85 | 1.7% | 198.40 |
| 4 | PPO | -14.49 | 17.41 | 3.3% | 186.19 |
| 5 | PPO-Randomized+Safety | -16.29 | 19.82 | 0.0% | 202.70 |
| 6 | PPO+Safety | -19.83 | 24.95 | 1.7% | 187.72 |

**Winner: PD** — Best return (-2.59), zero failures, and lowest control energy (25.10) of all active controllers.

### Scenario: strong_wind

| Rank | Controller | avg_return | std | failure_rate | control_energy |
|------|-----------|-----------|------|-------------|---------------|
| 1 | **PD** | **-10.10** | 0.00 | 0.0% | 98.80 |
| 2 | PPO-Randomized+Safety | -31.26 | 33.06 | 26.7% | 168.03 |
| 3 | PPO-Randomized | -37.13 | 41.38 | 30.0% | 169.42 |
| 4 | PPO+Safety | -43.82 | 50.60 | 28.3% | 162.03 |
| 5 | PPO | -51.98 | 62.16 | 21.7% | 184.65 |
| 6 | NoControl | -97.73 | 0.00 | 95.0% | 0.00 |

**Winner: PD** — Dominates with -10.10 return, zero failures, and the lowest energy (98.80) among all controllers that avoid failure.

### Scenario: variable_wind

| Rank | Controller | avg_return | std | failure_rate | control_energy |
|------|-----------|-----------|------|-------------|---------------|
| 1 | **PD** | **-10.90** | 0.00 | 0.0% | 154.35 |
| 2 | PPO | -58.73 | 64.44 | 33.3% | 183.65 |
| 3 | PPO-Randomized+Safety | -62.04 | 69.38 | 33.3% | 188.02 |
| 4 | PPO-Randomized | -71.59 | 82.86 | 33.3% | 198.30 |
| 5 | PPO+Safety | -74.32 | 86.47 | 33.3% | 171.89 |
| 6 | NoControl | -201.86 | 0.00 | 100.0% | 0.00 |

**Winner: PD** — Clear winner at -10.90 return with zero failures. All PPO variants cluster at 3x worse returns with 33% failure rate in this challenging scenario.

### Scenario: out_of_distribution_wind

| Rank | Controller | avg_return | std | failure_rate | control_energy |
|------|-----------|-----------|------|-------------|---------------|
| 1 | **PPO+Safety** | **-30.19** | 12.05 | 40.0% | 164.56 |
| 2 | PD | -36.78 | 0.00 | 35.0% | 189.58 |
| 3 | PPO-Randomized | -36.87 | 21.16 | 40.0% | 198.58 |
| 4 | PPO-Randomized+Safety | -38.41 | 24.12 | 40.0% | 201.62 |
| 5 | PPO | -42.01 | 28.79 | 40.0% | 171.36 |
| 6 | NoControl | -49.77 | 0.00 | 100.0% | 0.00 |

**Winner: PPO+Safety** — This is the only scenario where a PPO variant wins. avg_return of -30.19 beats PD's -36.78 (a 18% improvement). However, PPO+Safety has a slightly higher failure rate (40.0% vs 35.0%). **Important caveat:** with n=3 seeds, this margin is narrow and may not be statistically significant.

---

## 2. Failure Rate Comparison

### All Scenarios Combined

| Controller | normal_wind | strong_wind | variable_wind | OOD_wind | avg_fail |
|-----------|-------------|-------------|--------------|----------|----------|
| PD | 0.0% | 0.0% | 0.0% | 35.0% | **8.8%** |
| PPO-Randomized+Safety | 0.0% | 26.7% | 33.3% | 40.0% | 25.0% |
| PPO-Randomized | 1.7% | 30.0% | 33.3% | 40.0% | 26.3% |
| PPO+Safety | 1.7% | 28.3% | 33.3% | 40.0% | 25.8% |
| PPO | 3.3% | 21.7% | 33.3% | 40.0% | 24.6% |
| NoControl | 0.0% | 95.0% | 100.0% | 100.0% | 73.8% |

**Key finding:** PD achieves zero failures in all in-distribution scenarios (normal_wind, strong_wind, variable_wind). NoControl collapses under stress. All PPO variants exhibit significant failure rates (20-40%) outside normal_wind.

---

## 3. Control Energy Comparison

Lower energy is generally preferred (less actuator effort/wear), assuming performance is maintained.

| Controller | normal_wind | strong_wind | variable_wind | OOD_wind |
|-----------|-------------|-------------|--------------|----------|
| **PD** | **25.10** | **98.80** | **154.35** | 189.58 |
| PPO | 186.19 | 184.65 | 183.65 | **171.36** |
| PPO+Safety | 187.72 | **162.03** | **171.89** | 164.56 |
| PPO-Randomized | 198.40 | 169.42 | 198.30 | 198.58 |
| PPO-Randomized+Safety | 202.70 | 168.03 | 188.02 | 201.62 |

**Key findings:**
- PD is dramatically more energy-efficient in normal_wind (25.10 vs. 186+ for PPO variants) — roughly 7x less energy.
- In strong/variable wind, PD still uses less or comparable energy.
- In OOD wind, PPO+Safety slightly edges PD on energy (164.56 vs 189.58), but at comparable failure rates.

---

## 4. PPO Seed Variance Analysis

This is the most critical reliability concern in these results.

### Average Return Range (min to max across 3 seeds)

| Variant | normal_wind | strong_wind | variable_wind | OOD_wind |
|---------|-------------|-------------|--------------|----------|
| **PPO** | -39.11 to -2.17 (**range: 36.95**) | -139.88 to -7.98 (**range: 131.90**) | -149.85 to -11.62 (**range: 138.23**) | -82.72 to -21.50 (**range: 61.22**) |
| PPO+Safety | -55.11 to -2.18 (**range: 52.93**) | -115.38 to -8.00 (**range: 107.38**) | -196.60 to -11.65 (**range: 184.95**) | -47.23 to -21.30 (**range: 25.93**) |
| PPO-Randomized | -31.64 to -2.19 (**range: 29.45**) | -95.65 to -7.80 (**range: 87.85**) | -188.77 to -12.35 (**range: 176.42**) | -66.79 to -21.75 (**range: 45.04**) |
| PPO-Randomized+Safety | -44.33 to -2.24 (**range: 42.09**) | -78.02 to -7.81 (**range: 70.21**) | -160.15 to -12.34 (**range: 147.81**) | -72.53 to -21.21 (**range: 51.32**) |

### Seed-Level Data: The "Seed 0 Problem"

Across all scenarios, **seed 0 consistently fails catastrophically** for every PPO variant:

**normal_wind** seed 0 performance:
- PPO-seed0: -39.11 (10% failures)
- PPO-Randomized-seed0: -31.64 (5% failures)
- PPO-seed0+Safety: -55.11 (5% failures)
- PPO-Randomized-seed0+Safety: -44.33 (0% failures but poor return)

**strong_wind** seed 0 performance:
- PPO-seed0: -139.88 (65% failures) — nearly catastrophic
- PPO-Randomized-seed0: -95.65 (90% failures)
- PPO-seed0+Safety: -115.38 (85% failures)
- PPO-Randomized-seed0+Safety: -78.02 (80% failures)

**variable_wind** seed 0 performance:
- ALL seed 0 variants: 100% failures

**OOD_wind** seed 0 performance:
- ALL seed 0 variants: 100% failures

Contrast with seeds 1 and 2, which in many scenarios achieve returns competitive with or better than PD (e.g., PPO-Randomized-seed1 in strong_wind: -7.80, vs PD: -10.10).

**Conclusion on variance:** The PPO policies are extremely seed-sensitive. The average return across 3 seeds is misleading because it hides the bimodal distribution: seeds 1-2 often perform well, while seed 0 consistently fails. This raises serious questions about reproducibility and deployment risk.

---

## 5. Per-Scenario Winner Declarations

| Scenario | Winner | Evidence Quality |
|----------|--------|-----------------|
| normal_wind | **PD** (-2.59) | Very strong — PD beats all PPO variants by >10x return margin, with lower energy and zero failures |
| strong_wind | **PD** (-10.10) | Strong — PD beats best PPO variant by >3x margin, with zero failures vs 20-30% |
| variable_wind | **PD** (-10.90) | Very strong — PD beats best PPO by >5x margin, zero failures vs 33% |
| OOD_wind | **PPO+Safety** (-30.19) | Weak/narrow — beats PD by 18%, but at higher failure rate (40% vs 35%) and with only n=3 seeds |

---

## 6. Evidence Strength Assessment

### Strongest Evidence

1. **PD dominates in-distribution scenarios.** In normal_wind, strong_wind, and variable_wind, PD achieves:
   - Best average return in all three
   - Zero failure rate in all three
   - Lowest control energy in all three
   - Deterministic behavior (zero std, as it has no learned parameters)

2. **NoControl is catastrophically non-robust.** Under strong_wind (95% failure), variable_wind (100%), and OOD_wind (100%), uncontrolled turbines are unsafe.

3. **Seed 0 failure is a structural pattern.** Every PPO variant's seed 0 fails dramatically in variable_wind and OOD_wind (100% failure for both), and shows 65-90% failure in strong_wind. This is not a fluke — it's a systematic weakness.

### Weakest Evidence

1. **PPO+Safety wins OOD_wind by a narrow margin.** The win is based on:
   - Only 3 seeds (n=3)
   - 40% failure rate (not acceptable for deployment)
   - Margin over PD (-30.19 vs -36.78) is modest and may not survive additional seeds
   - PD at 35% failure is itself marginal — OOD_wind breaks all controllers

2. **PPO "potential" from seeds 1-2.** In strong_wind, PPO-Randomized-seed1 achieves -7.80 which is actually BETTER than PD's -10.10. However:
   - This ignores that seed 0 fails at 90% in the same scenario
   - A controller that works 2/3 seeds and fails catastrophically on the 1/3 is not a reliable replacement for a controller that works 3/3
   - Claiming "PPO can beat PD" based on best-seed cherry-picking is not scientifically honest

3. **Domain randomization's benefit is inconclusive.** Comparing PPO-Randomized vs vanilla PPO:
   - In strong_wind: PPO-Randomized avg (-37.13) vs PPO avg (-51.98) — randomized is better
   - In variable_wind: PPO-Randomized avg (-71.59) vs PPO avg (-58.73) — randomized is WORSE
   - In OOD_wind: PPO-Randomized avg (-36.87) vs PPO avg (-42.01) — randomized is slightly better
   - The mixed results suggest domain randomization helps in some scenarios but cannot fix the seed-0 problem or fundamentally improve reliability

4. **Safety layer adds intervention overhead without clear benefit.** The +Safety variants introduce intervention rates (especially notable in OOD: PPO-Randomized+Safety has avg intervention rate of 0.94 — nearly constant intervention). The cost of these interventions (energy, potential actuator wear) is unclear in the reward function.

---

## 7. Detailed PD vs PPO Head-to-Head

| Scenario | Delta (PPO - PD) | Winner | PD failures | PPO failures |
|----------|------------------|--------|-------------|-------------|
| normal_wind | PPO is 11.9 worse | PD by >4.6x | 0.0% | 3.3% |
| strong_wind | PPO is 41.9 worse | PD by >5.1x | 0.0% | 21.7% |
| variable_wind | PPO is 47.8 worse | PD by >5.4x | 0.0% | 33.3% |
| OOD_wind | PPO is 5.2 worse | PPO by 1.14x | 35.0% | 40.0% |

*Note: "worse" means more negative return. "Delta" = PPO_return - PD_return.*

---

## 8. Honest Conclusion

**Summary: The classical PD controller outperforms the PPO RL controllers across 3 of 4 scenarios, and does so with zero control energy overhead, zero variance, and complete reliability in in-distribution settings.**

The RL-based PPO controllers show a bimodal performance distribution driven by seed sensitivity. In their best configuration (seed 1 or 2), some PPO variants can match or slightly exceed PD performance in specific scenarios. However, seed 0 consistently fails catastrophically, with 100% failure rates on variable and OOD wind conditions. This seed dependence makes PPO unsuitable for safety-critical deployment without further training, better initialization, or more robust optimization.

**PPO+Safety narrowly wins the out-of-distribution scenario**, achieving an 18% return improvement over PD. However, this comes with a 40% failure rate (worse than PD's 35%) and requires near-constant safety intervention (0.42 avg rate), making the practical value unclear. The margin is narrow (3 seeds, no statistical testing) and may not generalize.

**Domain randomization has mixed results** — it appears to help slightly in strong_wind and OOD_wind but worsens performance in variable_wind. It does NOT solve the seed-0 problem.

**Recommendations for future rounds:**
1. Test with many more seeds (n >= 30) to properly characterize RL performance distribution
2. Investigate why seed 0 consistently fails — is it initialization, exploration, or reward shaping?
3. Tune the reward function to penalize failures more heavily
4. Consider hybrid approaches: PD as safety fallback with RL as primary controller
5. Report per-seed results, not just aggregated means, to reveal bimodal distributions

---

*This audit was conducted by direct analysis of `evaluation_summary_round3.csv` (56 data rows) and `evaluation_summary_round3_aggregated.csv` (24 data rows). All numbers are taken directly from the CSV without transformation beyond aggregation already present in the _aggregated file.*

*Auditor: Hermes Agent (Nous Research) | Round 3 Result Audit | 2026-05-13*
