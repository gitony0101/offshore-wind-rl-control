"""
Domain-randomized PPO training script for the OffshoreWind floating platform controller.

Identical to ``train_ppo.py`` in structure, but constructs the environment with
``randomized_training=True`` so that physical parameters (stiffness, damping,
mass, wind_std, wave_std) are re-sampled at every reset.  This encourages the
learned policy to generalise across a family of plants rather than over-fitting
to a single set of parameters.

Usage
-----
    python -m src.training.train_randomized_ppo --timesteps 50000 --scenario normal_wind

Smoke test:
    python -m src.training.train_randomized_ppo --smoke-test
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Ensure package root is on the import path.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.floating_platform_env import FloatingPlatformEnv  # noqa: E402


def make_env(scenario: str = "normal_wind"):
    """Return a zero-argument callable that constructs a FloatingPlatformEnv with domain randomization."""
    def _init() -> gym.Env:
        env = FloatingPlatformEnv(
            scenario=scenario,
            randomized_training=True,
        )
        return env
    return _init


def build_ppo(vec_env: DummyVecEnv, seed: int) -> PPO:
    """Construct a PPO model tuned for domain-randomized training."""
    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=1,
        tensorboard_log=None,
    )


def train_randomized(
    timesteps: int = 50000,
    seed: int = 42,
    scenario: str = "normal_wind",
    output_dir: Path = Path("results/"),
    log_dir: Path | None = None,
    verbose: int = 1,
    save_log: bool = False,
    model_name: str | None = None,
) -> PPO:
    """Run domain-randomized PPO training.

    Parameters
    ----------
    timesteps : int
        Total number of environment steps.
    seed : int
        Random seed.
    scenario : str
        Wind/wave scenario.
    output_dir : pathlib.Path
        Root directory for models and logs.
    log_dir : pathlib.Path | None
        Custom directory for VecMonitor log CSV output.
        If None and ``save_log`` is True, defaults to ``output_dir / "logs/{model_name}"``.
    verbose : int
        Verbosity level for PPO.
    save_log : bool
        If True, save VecMonitor training log CSV to results/logs/.
    model_name : str | None
        Base name for the saved model (default: ppo_randomized_{scenario}).

    Returns
    -------
    PPO
        The trained model.
    """
    print(f"\n{'='*60}")
    print(f"Randomized PPO Training — scenario: {scenario}, timesteps: {timesteps}, seed: {seed}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Prepare output directories
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if model_name is None:
        model_name = f"ppo_randomized_{scenario}"

    # Build vectorised environment with domain randomization enabled
    if save_log:
        if log_dir is None:
            log_dir = logs_dir / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        vec_env_log_dir = str(log_dir)
    else:
        vec_env_log_dir = None
    vec_env = DummyVecEnv([make_env(scenario)])
    vec_env.seed(seed)
    vec_env = VecMonitor(vec_env, vec_env_log_dir)

    # Build PPO
    model = build_ppo(vec_env, seed=seed)
    model.verbose = verbose

    # Prepare output directories
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Periodic checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=timesteps // 4,
        save_path=str(models_dir / f"checkpoint_randomized_{model_name}"),
        name_prefix=model_name,
    )

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    elapsed = time.time() - t0

    # Save final model
    model_path = models_dir / f"{model_name}.zip"
    model.save(str(model_path))
    print(f"\nModel saved: {model_path}")

    # Training log CSV from VecMonitor (written automatically to log_dir if set)
    csv_path = None
    if save_log and log_dir is not None:
        import glob
        csv_candidates = glob.glob(f"{log_dir}/*.csv")
        csv_path = Path(csv_candidates[0]) if csv_candidates else None
        if csv_path:
            print(f"Training log CSV saved: {csv_path}")

    # Extract episode info buffer from CSV if logging was enabled.
    ep_buffer = []
    if save_log and log_dir is not None:
        import glob as _glob
        csv_candidates = _glob.glob(f"{log_dir}/*.csv")
        for csv_path in csv_candidates:
            try:
                log_df = pd.read_csv(csv_path, comment="#")
                if "r" in log_df.columns:
                    ep_buffer = [{"r": float(r)} for r in log_df["r"].values]
                    break
            except Exception:
                continue
    # Summary
    print(f"\n{'─'*40}")
    print(f"Training summary (randomized)")
    print(f"{'─'*40}")
    print(f"  Total timesteps : {timesteps}")
    if ep_buffer:
        mean_reward = sum(d["r"] for d in ep_buffer) / len(ep_buffer)
        print(f"  Mean episode reward : {mean_reward:.4f}")
        print(f"  Episodes logged    : {len(ep_buffer)}")
    else:
        print(f"  No episode info captured (too few timesteps).")
    print(f"  Training time      : {elapsed:.1f} s")
    print(f"  Model path         : {model_path}")
    if csv_path:
        print(f"  Log CSV path       : {csv_path}")
    else:
        print(f"  (Training log CSV not written; use --save-log)")
    print("")

    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent with domain randomization on the OffshoreWind environment.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default="normal_wind",
        choices=["normal_wind", "strong_wind", "variable_wind", "out_of_distribution_wind"],
        help="Wind/wave scenario preset (default: normal_wind).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/",
        help="Directory for saving models and logs (default: results/).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override timesteps to 1000 and reduce verbosity for a quick sanity check.",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        default=False,
        help="Save VecMonitor training log CSV to results/logs/.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Base name for saved model and logs (default: ppo_randomized_{scenario}).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom directory for VecMonitor training log CSV output.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> PPO:
    """CLI entry-point for randomized training."""
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    timesteps = args.timesteps if not args.smoke_test else 1000
    verbose = 0 if args.smoke_test else 1

    # Seed global numpy
    np.random.seed(args.seed)

    model = train_randomized(
        timesteps=timesteps,
        seed=args.seed,
        scenario=args.scenario,
        output_dir=output_dir,
        log_dir=Path(args.log_dir) if args.log_dir else None,
        verbose=verbose,
        save_log=args.save_log,
        model_name=args.model_name,
    )

    return model


if __name__ == "__main__":
    main()
