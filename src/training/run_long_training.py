"""
Long-horizon multi-seed training runner for Round 3.

Sequentially trains:
  - Standard PPO: seeds 0, 1, 2  (500k timesteps each)
  - Randomized PPO: seeds 0, 1, 2  (500k timesteps each)

Saves models to results/models/ with naming:
  ppo_normal_500k_seed{0,1,2}.zip
  ppo_randomized_500k_seed{0,1,2}.zip

Usage
-----
    python -m src.training.run_long_training --timesteps 500000 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_training(
    script: str,
    timesteps: int,
    seed: int,
    model_name: str,
    save_log: bool = True,
    output_dir: str = "results",
) -> bool:
    """Run a single training job as a subprocess."""
    cmd = [
        sys.executable, "-m", script,
        "--timesteps", str(timesteps),
        "--seed", str(seed),
        "--model-name", model_name,
        "--output-dir", output_dir,
    ]
    if save_log:
        cmd.append("--save-log")

    print(f"\n{'='*60}")
    print(f"  Starting: {model_name}")
    print(f"  Script: {script}")
    print(f"  Timesteps: {timesteps}, Seed: {seed}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0

    if result.returncode == 0:
        print(f"\n  OK: {model_name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n  FAILED: {model_name} (exit code {result.returncode}, {elapsed:.1f}s)")
        return False


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Round 3: Long multi-seed training runner.")
    parser.add_argument(
        "--timesteps", type=int, default=500000,
        help="Training timesteps per seed (default: 500000).",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0, 1, 2],
        help="Random seeds to use (default: 0 1 2).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory (default: results).",
    )
    args = parser.parse_args(argv)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    jobs = []

    # Standard PPO jobs
    for seed in args.seeds:
        model_name = f"ppo_normal_500k_seed{seed}"
        jobs.append(("src.training.train_ppo", model_name, seed))

    # Randomized PPO jobs
    for seed in args.seeds:
        model_name = f"ppo_randomized_500k_seed{seed}"
        jobs.append(("src.training.train_randomized_ppo", model_name, seed))

    print(f"\nRound 3 Training Plan")
    print(f"  Timesteps per run: {args.timesteps}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Total training runs: {len(jobs)}")
    print(f"  Estimated time: ~{len(jobs) * 3:0d} minutes (approx 3 min per run on CPU)\n")

    results = []
    for script, model_name, seed in jobs:
        ok = run_training(
            script=script,
            timesteps=args.timesteps,
            seed=seed,
            model_name=model_name,
            save_log=True,
            output_dir=args.output_dir,
        )
        results.append((model_name, ok))

    # Summary
    print(f"\n\n{'='*60}")
    print(f"  Round 3 Training Summary")
    print(f"{'='*60}")
    for name, ok in results:
        status = "OK" if ok else "FAILED"
        print(f"  [{status}] {name}")
    
    passed = sum(1 for _, ok in results if ok)
    print(f"\n  {passed}/{len(results)} runs completed successfully.\n")


if __name__ == "__main__":
    main()
