"""Environment smoke test for FloatingPlatformEnv.

Runs a quick sanity check on the custom Gymnasium environment:
- Creates environments for all scenarios
- Verifies reset() returns valid observation
- Steps 100 timesteps without error
- Checks observation/action space shapes
- Verifies done conditions
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_project_root = str(Path(__file__).parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.envs.floating_platform_env import FloatingPlatformEnv


def _get_env_and_obs(scenario="normal_wind"):
    """Helper that returns a reset environment and observation."""
    env = FloatingPlatformEnv(scenario=scenario)
    obs, info = env.reset(seed=42)
    return env, obs


def test_env_reset(scenario="normal_wind"):
    """Test that reset returns a valid observation."""
    env, obs = _get_env_and_obs(scenario)
    assert isinstance(obs, np.ndarray), "Observation should be numpy array"
    assert obs.shape == (4,), f"Expected shape (4,) but got {obs.shape}"
    assert np.all(np.isfinite(obs)), "Observation contains non-finite values"
    print(f"  PASS: reset() for {scenario} -- obs shape={obs.shape}")


def test_env_steps(n_steps=100):
    """Test that stepping works without errors."""
    env = FloatingPlatformEnv()
    obs, _ = env.reset(seed=42)
    rng = np.random.RandomState(42)

    for _ in range(n_steps):
        action = np.array([rng.uniform(-1.0, 1.0)])
        obs_next, reward, terminated, truncated, info = env.step(action)
        assert obs_next.shape == (4,), f"bad obs shape: {obs_next.shape}"
        assert np.isfinite(reward), "Reward is not finite"
        assert isinstance(terminated, (bool, np.bool_)), "terminated should be bool"
        assert isinstance(truncated, (bool, np.bool_)), "truncated should be bool"
        if terminated:
            assert abs(obs_next[0]) > 0.28, f"Safety termination but theta={obs_next[0]}"
            break

    print(f"  PASS: {n_steps} steps completed without error")


def test_all_scenarios():
    """Smoke test all wind scenarios."""
    scenarios = ["normal_wind", "strong_wind", "variable_wind", "out_of_distribution_wind"]
    for s in scenarios:
        env, obs = _get_env_and_obs(scenario=s)
        obs, _, term, trunc, _ = env.step(np.array([0.0]))
        print(f"    {s}: term={term}, trunc={trunc}")
        # Reset and step to 100
        _run_steps_with_env(env, n_steps=100)


def test_all_scenarios_fresh():
    """Smoke test all wind scenarios with fresh envs."""
    scenarios = ["normal_wind", "strong_wind", "variable_wind", "out_of_distribution_wind"]
    for s in scenarios:
        env, obs = _get_env_and_obs(scenario=s)
        obs, _, term, trunc, _ = env.step(np.array([0.0]))
        print(f"    {s}: term={term}, trunc={trunc}")
        _run_steps_with_env(env, n_steps=100)


def _run_steps_with_env(env, n_steps=100):
    """Test stepping with a pre-created environment."""
    obs, _ = env.reset(seed=42)
    rng = np.random.RandomState(42)

    for _ in range(n_steps):
        action = np.array([rng.uniform(-1.0, 1.0)])
        obs_next, reward, terminated, truncated, info = env.step(action)
        assert obs_next.shape == (4,)
        assert np.isfinite(reward)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        if terminated:
            break

    print(f"  PASS: {n_steps} steps completed without error (scenario={env.scenario})")


def test_observation_space():
    """Test observation space bounds."""
    env = FloatingPlatformEnv()
    space = env.observation_space
    assert space.shape == (4,), f"Expected (4,) but got {space.shape}"
    assert space.dtype == np.float32, f"Expected float32 but got {space.dtype}"
    print(f"  PASS: observation space shape={space.shape}, dtype={space.dtype}")


def test_action_space():
    """Test action space bounds."""
    env = FloatingPlatformEnv()
    space = env.action_space
    assert space.shape == (1,), f"Expected (1,) but got {space.shape}"
    assert space.dtype == np.float32, f"Expected float32 but got {space.dtype}"
    assert space.low[0] == -1.0 and space.high[0] == 1.0, "Action bounds should be [-1, 1]"
    print(f"  PASS: action space shape={space.shape}, low={space.low}, high={space.high}")


def test_seed_reproducibility():
    """Test that same seed gives same trajectory."""
    env = FloatingPlatformEnv()

    obs1, _ = env.reset(seed=123)
    traj1 = []
    for _ in range(50):
        act = np.array([0.0])
        obs1, _, term1, trunc1, _ = env.step(act)
        traj1.append(obs1.copy())
        if term1 or trunc1:
            break

    obs2, _ = env.reset(seed=123)
    traj2 = []
    for _ in range(50):
        act = np.array([0.0])
        obs2, _, term2, trunc2, _ = env.step(act)
        traj2.append(obs2.copy())
        if term2 or trunc2:
            break

    assert len(traj1) == len(traj2), f"Length mismatch: {len(traj1)} vs {len(traj2)}"
    for i, (o1, o2) in enumerate(zip(traj1, traj2)):
        assert np.allclose(o1, o2, atol=1e-6), f"Mismatch at step {i}"
    print("  PASS: seed reproducibility confirmed")


def test_info_dict():
    """Test that info dict contains required keys."""
    env = FloatingPlatformEnv()
    obs, _ = env.reset(seed=42)
    _, _, terminated, truncated, info = env.step(np.array([0.5]))
    required_keys = ["theta", "theta_dot", "wind", "wave", "action"]
    for key in required_keys:
        assert key in info, f"Missing info key: {key}"
    print(f"  PASS: info dict has required keys: {required_keys}")


if __name__ == "__main__":
    print("=" * 60)
    print("Environment Smoke Tests")
    print("=" * 60)

    test_observation_space()
    test_action_space()
    test_env_reset()
    test_env_steps()
    test_seed_reproducibility()
    test_info_dict()
    test_all_scenarios_fresh()

    print("=" * 60)
    print("All smoke tests passed.")
    print("=" * 60)