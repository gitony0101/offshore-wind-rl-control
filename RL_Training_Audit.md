# RL Training Audit

## Training Reproducibility
The training scripts support full reproducibility through:
- CLI `--seed` argument (default 42) that seeds:
  - NumPy global state (`np.random.seed(args.seed)`)
  - Stable-Baselines3 PPO constructor (`seed=seed`)
  - Vectorized environment (`vec_env.seed(seed)`)
  - Environment internal randomization via `self.np_random`
- Deterministic environment resets when seeded
- Identical hyperparameters across runs when seeds match

To recreate the six 500K models (PPO and PPO-Randomized for seeds 0,1,2):
```bash
# Standard PPO
python -m src.training.train_ppo --timesteps 500000 --seed 0 --scenario normal_wind
python -m src.training.train_ppo --timesteps 500000 --seed 1 --scenario normal_wind
python -m src.training.train_ppo --timesteps 500000 --seed 2 --scenario normal_wind

# Domain-randomized PPO
python -m src.training.train_randomized_ppo --timesteps 500000 --seed 0 --scenario normal_wind
python -m src.training.train_randomized_ppo --timesteps 500000 --seed 1 --scenario normal_wind
python -m src.training.train_randomized_ppo --timesteps 500000 --seed 2 --scenario normal_wind
```

Model files are saved unambiguously to `results/models/` with naming patterns like:
- `ppo_normal_500k_seed0.zip`
- `ppo_randomized_500k_seed0.zip`

## Logging Integrity
Both scripts properly save training logs when `--save-log` is used:
- Utilize `VecMonitor` wrapper which automatically writes Monitor CSV logs
- Logs saved to `results/logs/{model_name}/monitor.csv` (or custom `--log-dir`)
- CSV format includes standard columns: `r` (episode return), `l` (episode length), `t` (timestamp)
- Verified existence of monitor.csv files in results/logs/ for various training runs

The `plot_results.py` script can successfully parse these logs by:
- Looking for files matching patterns like `*_training_log.csv` or `monitor.csv`
- Reading CSV with comment='#' to skip VecMonitor header lines
- Extracting reward column from available columns (checks for 'r', 'ep_rew_mean', etc.)

## Seed Handling
Seeds are properly applied to all relevant sources:
1. **NumPy**: `np.random.seed(args.seed)` in main()
2. **Environment**: Via `DummyVecEnv.seed(seed)` which propagates to individual envs
3. **Stable-Baselines3 PPO**: Direct seed parameter in constructor
4. **Environment internal randomness**: Uses `self.np_random` seeded by VecEnv

This ensures that with identical seeds, environment resets, action sampling, and parameter randomization (for domain-randomized version) produce identical sequences.

## Domain Randomization Integrity
The domain-randomized PPO training properly randomizes intended parameters:
- When `randomized_training=True`, the `FloatingPlatformEnv.reset()` method samples:
  - Stiffness: uniform(0.8, 2.2) [±~47% from default 1.5]
  - Damping: uniform(0.2, 0.8) [±~60% from default 0.5]
  - Mass: uniform(0.5, 1.5) [±~50% from default 1.0]
  - Wind std: uniform(0.1, 0.6) [for normal_wind scenario]
  - Wave std: uniform(0.05, 0.4) [for normal_wind scenario]

These ranges are appropriate for 500K timesteps training - wide enough to encourage robustness but not so wide as to make learning impossible. The randomization occurs at every `reset()`, meaning each episode presents a different plant dynamics configuration.

## Training Risks Identified
1. **Code Duplication**: ~80% of code is duplicated between `train_ppo.py` and `train_randomized_ppo.py`, increasing maintenance burden
2. **Timestep Confusion**: Default timesteps is 50,000 (50K) but 500K models exist in results - users might accidentally undertrain
3. **Limited Hyperparameter Documentation**: No explanation of why specific PPO hyperparameters were chosen
4. **No GPU Option**: Training hard-coded to CPU; no command-line option to utilize GPU if available
5. **Inconsistent Default Model Names**: Standard PPO uses `ppo_{scenario}_{timesteps}` while randomized uses `ppo_randomized_{scenario}` (omits timesteps)

## Recommended Fixes
1. **Extract Common Training Logic**: Create a shared `train_ppo_base()` function that both scripts call, passing a `randomize` flag
2. **Update Default Timesteps**: Change default from 50000 to 500000 to match the models presented in results, or at least update documentation to be explicit
3. **Add Hyperparameter Comments**: Document reasoning behind chosen PPO hyperparameters (e.g., "These values work well for continuous control tasks similar to MuJoCo benchmarks")
4. **Add Device Argument**: Include `--device` option (auto/cuda/cpu) passed to PPO constructor
5. **Standardize Model Naming**: Make both scripts include timesteps in default model name for consistency (e.g., `ppo_randomized_{scenario}_{timesteps}`)
6. **Add Validation**: Check that `--timesteps` is positive and provide warning if unusually low/high
7. **Improve Log Verification**: Add explicit test in plot_results.py to verify it can find and parse logs from both training scripts

## Conclusion
The training protocol is fundamentally sound for comparative evaluation. Both algorithms receive identical computational budgets (same timesteps), use identical hyperparameters, and can be run with identical seeds for fair comparison. The domain randomization is properly implemented and constrained to reasonable ranges. With the recommended fixes addressing maintainability and clarity issues, the reproducibility and reliability of the RL training pipeline would be significantly enhanced.