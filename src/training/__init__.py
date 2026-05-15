"""
Training module for the OffshoreWind floating platform RL controller.

Exports the main training functions and factory helpers used to train PPO
agents on the ``FloatingPlatformEnv`` environment.

Typical usage:

    from src.training import train, train_randomized

    # Standard training
    model = train(timesteps=50000, scenario="normal_wind", seed=42)

    # Domain-randomized training
    model = train_randomized(timesteps=50000, scenario="strong_wind", seed=42)
"""

from src.training.train_ppo import (
    build_ppo,
    make_env,
    main as train,
    parse_args,
    train as train_standard,
)
from src.training.train_randomized_ppo import (
    main as train_randomized,
)

__all__ = [
    "build_ppo",
    "make_env",
    "parse_args",
    "train",
    "train_standard",
    "train_randomized",
]
