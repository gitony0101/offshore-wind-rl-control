"""
Standalone PPO training script for the OffshoreWind floating platform controller.

Trains a Stable-Baselines3 PPO agent (MlpPolicy) on a single scenario of the
FloatingPlatformEnv environment.  Supports reproducibility via seeds, logging
via CSV and TensorBoard, and a ``--smoke-test`` mode for quick validation.

Usage
-----
Typical training:

    python -m src.training.train_ppo --timesteps 50000 --scenario normal_wind

Smoke test (1000 steps, minimal output):

    python -m src.training.train_ppo --smoke-test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import csv

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# Ensure package root is on the import path so the environment module is
# discoverable regardless of how this script is invoked.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.envs.floating_platform_env import FloatingPlatformEnv  # noqa: E402


def make_env(scenario: str = "normal_wind"):
    """Return a zero-argument callable that constructs a FloatingPlatformEnv."""
    def _init() -> gym.Env:
        env = FloatingPlatformEnv(scenario=scenario)
        return env
    return _init


def build_ppo(vec_env: DummyVecEnv, seed: int) -> PPO:
    """Construct a PPO model with sensible defaults for the floating platform environment."""
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


def train(
    timesteps: int = 50000,
    seed: int = 42,
    scenario: str = "normal_wind",
    output_dir: Path = Path("results/"),
    log_dir: Path | None = None,
    verbose: int = 1,
    save_log: bool = False,
    model_name: str | None = None,
) -> PPO:
    """Run PPO training and save the model and log CSV.

    Parameters
    ----------
    timesteps : int
        Total number of environment steps to train for.
    seed : int
        Random seed for reproducibility.
    scenario : str
        Which wind/wave scenario preset to use.
    output_dir : pathlib.Path
        Directory root for saving models and logs.
    log_dir : pathlib.Path | None
        Custom directory for VecMonitor log CSV output.
        If None and ``save_log`` is True, defaults to ``output_dir / "logs/{model_name}"``.
    verbose : int
        Verbosity level passed to the PPO constructor.
    save_log : bool
        If True, VecMonitor writes episode-level CSV logs to ``log_dir``.
    model_name : str | None
        Name for the saved model (without .zip extension).
        Defaults to ``ppo_{scenario}_{timesteps}``.

    Returns
    -------
    PPO
        The trained model.
    """
    if model_name is None:
        model_name = f"ppo_{scenario}_{timesteps}"
    print(f"\n{'='*60}")
    print(f"PPO Training — scenario: {scenario}, timesteps: {timesteps}, seed: {seed}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # Prepare output directories
    models_dir = output_dir / "models"
    logs_dir = output_dir / "logs"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Build vectorised environment with VecMonitor for CSV logging
    if save_log:
        if log_dir is None:
            log_dir = logs_dir / model_name
        log_dir.mkdir(parents=True, exist_ok=True)
        vec_env_log_dir = str(log_dir)
    else:
        vec_env_log_dir = None
    vec_env = DummyVecEnv([make_env(scenario=scenario)])
    vec_env.seed(seed)
    vec_env = VecMonitor(vec_env, vec_env_log_dir)

    # Build PPO model
    model = build_ppo(vec_env, seed=seed)
    model.verbose = verbose

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=timesteps // 4,
        save_path=str(models_dir / f"checkpoint_{scenario}"),
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

    # VecMonitor writes Monitor CSV automatically when log_dir is provided.
    # The CSV has columns: r (return), l (length), t (timestamp).
    if save_log and log_dir is not None:
        import glob

        csv_candidates = glob.glob(f"{log_dir}/*.csv")
        if csv_candidates:
            csv_path = Path(csv_candidates[0])
            print(f"Training log CSV saved: {csv_path}")
        else:
            # Fallback: write CSV from VecMonitor buffer
            csv_path = logs_dir / f"{model_name}_training_log.csv"
            ep_buffer = getattr(vec_env, "ep_info_buffer", None) or []
            if ep_buffer:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["r", "l", "t"])
                    for info in ep_buffer:
                        writer.writerow(
                            [info.get("r", 0), info.get("l", 0), info.get("t", 0)]
                        )
                print(f"Training log CSV saved: {csv_path}")
    else:
        csv_path = None

    # Summary
    print(f"\n{'─'*40}")
    print(f"Training summary")
    print(f"{'─'*40}")
    print(f"  Total timesteps : {timesteps}")
    # VecMonitor may have stored a MonitorCSVLogger; fall back gracefully.
    ep_buffer = []
    if csv_path and Path(csv_path).exists():
        try:
            with open(csv_path, "r") as f:
                lines = f.readlines()
            for line in lines[3:]:  # skip VecMonitor header comment lines
                parts = line.strip().split(",")
                if len(parts) >= 1 and parts[0].replace(".", "").replace("-", "").isnumeric():
                    ep_buffer.append({"r": float(parts[0])})
        except Exception:
            pass
    if ep_buffer and len(ep_buffer) > 0:
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
        description="Train a PPO agent on the OffshoreWind floating platform environment.",
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
        "--model-name",
        type=str,
        default=None,
        help="Base name for saved model and logs (default: ppo_{scenario}_{timesteps}).",
    )
    parser.add_argument(
        "--save-log",
        action="store_true",
        default=False,
        help="Save VecMonitor training log CSV to results/logs/.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Custom directory for VecMonitor log CSV output (default: <output-dir>/logs/<model-name>).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> PPO:
    """CLI entry-point for training."""
    args = parse_args(argv)

    output_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir) if args.log_dir else None
    timesteps = args.timesteps if not args.smoke_test else 1000
    verbose = 0 if args.smoke_test else 1

    # Seed global numpy for extra determinism
    np.random.seed(args.seed)

    model = train(
        timesteps=timesteps,
        seed=args.seed,
        scenario=args.scenario,
        output_dir=output_dir,
        log_dir=log_dir,
        verbose=verbose,
        save_log=args.save_log,
        model_name=args.model_name,
    )

    return model


if __name__ == "__main__":
    main()
