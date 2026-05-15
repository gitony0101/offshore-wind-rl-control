"""
Floating Offshore Wind Platform Pitch Control Environment.

A simplified single-axis pitch dynamics Gymnasium environment for training
reinforcement learning agents to stabilise floating offshore wind platforms.

Engineering context
-------------------
Floating offshore wind turbines (e.g. on semi-submersible or spar platforms)
experience platform pitch motion caused by aerodynamic forces from the wind
and hydrodynamic forces from ocean waves. Excessive pitch angles can lead to
structural fatigue, reduced power quality, and even catastrophic failure.
This environment provides an abstract, low-dimensional representation of that
problem so that RL policies can be rapidly prototyped without a full-fidelity
simulator (OpenFAST, MoorPy, FLORIS, …).

The state vector captures the platform pitch angle (theta, in radians), its
angular velocity (theta_dot, in rad/s), and two scalar proxies for wind and
wave disturbance forces.  The single continuous action (normalised to [-1, 1])
represents a combined control effort — in a real system this could map to
blade-pitch commands, generator-torque set-points, or individual blade control.

This is intentionally simplified: there are no tower dynamics, mooring-line
compliance, or multi-body effects.  It is meant for fast iteration and
algorithm development.

Usage
-----
>>> import gymnasium as gym
>>> from src.envs.floating_platform_env import FloatingPlatformEnv
>>> env = FloatingPlatformEnv(scenario="strong_wind")
>>> obs, info = env.reset()
>>> action = env.action_space.sample()
>>> obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FloatingPlatformEnv(gym.Env):
    """A single-axis pitch control environment for a floating wind platform.

    Parameters
    ----------
    stiffness : float, default 1.5
        Restoring torque coefficient (N·m/rad).  In a real floating platform
        this represents the combined hydrostatic restoring stiffness from
        buoyancy and mooring pretension that resists pitch motion.
    damping : float, default 0.5
        Viscous damping coefficient (N·m·s/rad).  Represents hydrodynamic
        radiation damping and structural energy dissipation.
    mass : float, default 1.0
        Generalised mass (kg) of the pitch degree of freedom.  In a full
        multi-body model this would be the coupled mass + added mass.
    dt : float, default 0.05
        Simulation time-step in seconds.  Governs the frequency at which
        the RL agent receives observations and issues actions.
    max_steps : int, default 1000
        Maximum number of steps per episode.  Acts as a time horizon
        (``max_steps * dt`` seconds of simulated time).
    safety_threshold : float, default 0.3
        Absolute pitch angle limit (radians) at which the episode terminates
        with a safety violation.  0.3 rad ≈ 17 degrees — beyond this angle
        structural damage or loss of aerodynamic efficiency is likely in a
        real turbine.
    action_gain : float, default 0.5
        Maps the normalised action ∈ [-1, 1] to a physical control force
        (N·m).  Larger gain means more control authority but risks
        overshooting if the policy is aggressive.
    wind_mean : float, default 0.0
        Mean of the normal distribution used to sample the stochastic wind
        disturbance proxy at each step (N·m).
    wind_std : float, default 0.3
        Standard deviation of the wind disturbance proxy (N·m).
    wave_std : float, default 0.2
        Standard deviation of the wave disturbance proxy (N·m).  Wave
        disturbances are zero-mean by default but can be biased via
        ``randomized_training``.
    scenario : str, default "normal_wind"
        Wind / wave scenario preset.  Supported values:
        - ``"normal_wind"``: uses the default ``wind_std`` and ``wave_std``.
        - ``"strong_wind"``: ``wind_std=0.6``, ``wave_std=0.4``.
        - ``"variable_wind"``: wind mean ramps linearly from 0 at step 0 to
          0.5 at ``max_steps``.
        - ``"out_of_distribution_wind"``: ``wind_std=0.9``, ``wave_std=0.7``.
          Intended for robustness testing.
    randomized_training : bool, default False
        When ``True``, randomise the physical parameters (``stiffness``,
        ``damping``, ``mass``) at every ``reset()``, as well as the
        ``wind_std`` / ``wave_std`` scales.  Useful for domain-randomised
        training to improve sim-to-real transfer.
    """

    metadata = {"render_modes": []}

    _SCENARIOS: dict[str, dict[str, float]] = {
        "normal_wind": {},
        "strong_wind": {"wind_std": 0.6, "wave_std": 0.4},
        "variable_wind": {},
        "out_of_distribution_wind": {"wind_std": 0.9, "wave_std": 0.7},
    }

    def __init__(
        self,
        stiffness: float = 1.5,
        damping: float = 0.5,
        mass: float = 1.0,
        dt: float = 0.05,
        max_steps: int = 1000,
        safety_threshold: float = 0.3,
        action_gain: float = 0.5,
        wind_mean: float = 0.0,
        wind_std: float = 0.3,
        wave_std: float = 0.2,
        scenario: str = "normal_wind",
        randomized_training: bool = False,
    ):
        super().__init__()

        # -- defaults (stored so we can randomise around them) --
        self._default_stiffness = stiffness
        self._default_damping = damping
        self._default_mass = mass
        self._default_wind_mean = wind_mean
        self._default_wind_std = wind_std
        self._default_wave_std = wave_std

        self.dt = dt
        self.max_steps = max_steps
        self.safety_threshold = safety_threshold
        self.action_gain = action_gain
        self.randomized_training = randomized_training

        self.scenario = scenario

        # Apply scenario overrides to the wind / wave scale parameters.
        overrides = self._SCENARIOS.get(scenario, {})
        self.wind_mean = wind_mean
        self.wind_std = overrides.get("wind_std", wind_std)
        self.wave_std = overrides.get("wave_std", wave_std)

        # Effective (possibly randomised) physical parameters — reset() will
        # overwrite these if randomized_training is True.
        self.stiffness = stiffness
        self.damping = damping
        self.mass = mass

        # -- spaces --
        # Observation: [theta, theta_dot, wind_disturbance, wave_disturbance]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32,
        )

        # Action: single continuous value normalised to [-1, 1].
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Internal state
        self._current_step: int = 0
        self._terminated_by_safety: bool = False

    # -------------------------------------------------------------------------
    # helpers
    # -------------------------------------------------------------------------

    def _sample_disturbances(self) -> Tuple[float, float]:
        """Sample wind and wave disturbance proxies for the current step.

        Returns
        -------
        wind_force : float
            Stochastic wind-force proxy (N·m).
        wave_force : float
            Stochastic wave-force proxy (N·m).
        """
        rng = self.np_random

        # In "variable_wind" the mean ramps linearly over the episode.
        if self.scenario == "variable_wind":
            progress = min(self._current_step / max(self.max_steps, 1), 1.0)
            wind_mean = progress * 0.5  # ramps 0 → 0.5
        else:
            wind_mean = self.wind_mean

        wind_force = wind_mean + self.wind_std * rng.standard_normal()
        wave_force = self.wave_std * rng.standard_normal()

        return float(wind_force), float(wave_force)

    def _compute_reward(
        self, theta: float, theta_dot: float, control_force: float
    ) -> float:
        """Compute the step reward.

        The reward penalises:
        - Pitch-angle deviation (quadratic) — we want the platform upright.
        - Angular velocity (quadratic) — rapid motion is undesirable.
        - Control effort (quadratic) — energy cost of actuation.
        - Approaching the safety boundary with a large bonus penalty.
        """
        r = (
            -1.0 * theta * theta
            - 0.5 * theta_dot * theta_dot
            - 0.1 * control_force * control_force
        )

        safety_penalty = 0.0
        if abs(theta) > self.safety_threshold * 0.9:
            safety_penalty = -10.0

        return r + safety_penalty

    # -------------------------------------------------------------------------
    # Gymnasium API
    # -------------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance the simulation by one time-step.

        Parameters
        ----------
        action : np.ndarray of shape (1,)
            Normalised control action in [-1, 1].

        Returns
        -------
        observation : np.ndarray of shape (4,)
            ``[theta, theta_dot, wind_disturbance, wave_disturbance]``.
        reward : float
        terminated : bool
            True if the safety threshold has been exceeded.
        truncated : bool
            True if the episode reached ``max_steps``.
        info : dict
            Diagnostic quantities including current state, disturbances, action,
            and whether termination was caused by a safety violation.
        """
        # Unpack action
        action_scalar = float(np.clip(action, -1.0, 1.0)[0])
        control_force = action_scalar * self.action_gain

        # Current state
        theta = self._state[0]
        theta_dot = self._state[1]

        # Sample disturbances
        wind_force, wave_force = self._sample_disturbances()

        # Integrate dynamics (Euler)
        # theta_next = theta + theta_dot * dt
        # theta_dot_next = theta_dot + dt * (
        #     wind_force + wave_force + control_force
        #     - damping * theta_dot - stiffness * theta
        # ) / mass
        net_force = (
            wind_force
            + wave_force
            + control_force
            - self.damping * theta_dot
            - self.stiffness * theta
        )
        theta_next = theta + theta_dot * self.dt
        theta_dot_next = theta_dot + self.dt * net_force / self.mass

        # Compute reward
        reward = self._compute_reward(theta_next, theta_dot_next, control_force)

        # Update internal state
        self._state[0] = theta_next
        self._state[1] = theta_dot_next
        self._state[2] = wind_force
        self._state[3] = wave_force

        # Done conditions
        terminated = abs(theta_next) > self.safety_threshold
        self._terminated_by_safety = terminated
        self._current_step += 1
        truncated = self._current_step >= self.max_steps

        # Observation
        obs = np.array(
            [
                self._state[0],
                self._state[1],
                self._state[2],
                self._state[3],
            ],
            dtype=np.float32,
        )

        info: Dict[str, Any] = {
            "theta": float(self._state[0]),
            "theta_dot": float(self._state[1]),
            "wind": float(wind_force),
            "wave": float(wave_force),
            "action": float(action_scalar),
            "terminated_by_safety": terminated,
        }

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Initial conditions:
        - ``theta`` is drawn uniformly from [-0.1, 0.1] (≈ ±5.7 degrees).
        - ``theta_dot`` is drawn uniformly from [-0.05, 0.05] (rad/s).

        When ``randomized_training`` is True the physical parameters
        (``stiffness``, ``damping``, ``mass``) and disturbance scales
        (``wind_std``, ``wave_std``) are also randomised to encourage robust
        policies.
        """
        super().reset(seed=seed)

        self._current_step = 0
        self._terminated_by_safety = False

        rng = self.np_random

        # Randomise physical parameters if in domain-randomisation mode
        if self.randomized_training:
            self.stiffness = float(rng.uniform(0.8, 2.2))
            self.damping = float(rng.uniform(0.2, 0.8))
            self.mass = float(rng.uniform(0.5, 1.5))
            self.wind_std = float(rng.uniform(0.1, self._default_wind_std * 2.0))
            self.wave_std = float(rng.uniform(0.05, self._default_wave_std * 2.0))
        else:
            overrides = self._SCENARIOS.get(self.scenario, {})
            self.stiffness = self._default_stiffness
            self.damping = self._default_damping
            self.mass = self._default_mass
            self.wind_std = overrides.get("wind_std", self._default_wind_std)
            self.wave_std = overrides.get("wave_std", self._default_wave_std)
            self.wind_mean = self._default_wind_mean

        self._state = np.zeros(4, dtype=np.float32)
        self._state[0] = float(rng.uniform(-0.1, 0.1))   # theta
        self._state[1] = float(rng.uniform(-0.05, 0.05))  # theta_dot

        # Sample initial disturbances so the observation is fully populated.
        wind_force, wave_force = self._sample_disturbances()
        self._state[2] = wind_force
        self._state[3] = wave_force

        obs = self._state.copy()

        return obs, {}

    # -------------------------------------------------------------------------
    # optional: convenience wrapper to expose scenario info
    # -------------------------------------------------------------------------

    def get_scenario_info(self) -> Dict[str, Any]:
        """Return a dict summarising the current scenario configuration."""
        return {
            "scenario": self.scenario,
            "stiffness": self.stiffness,
            "damping": self.damping,
            "mass": self.mass,
            "wind_mean": self.wind_mean,
            "wind_std": self.wind_std,
            "wave_std": self.wave_std,
            "safety_threshold": self.safety_threshold,
            "action_gain": self.action_gain,
            "randomized_training": self.randomized_training,
        }


if __name__ == "__main__":
    # Quick smoke-test: run a 10-step episode with a random policy under three
    # different scenarios.
    import sys

    scenarios = ["normal_wind", "strong_wind", "variable_wind", "out_of_distribution_wind"]

    passed = 0
    for scenario in scenarios:
        print(f"\n{'='*50}")
        print(f"Testing scenario: {scenario}")
        print("="*50)

        env = FloatingPlatformEnv(scenario=scenario)
        obs, info = env.reset(seed=42)
        assert obs.shape == (4,), f"Expected obs shape (4,), got {obs.shape}"

        term, trunc = False, False
        for step_idx in range(10):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            print(
                f"  step={step_idx:2d}  "
                f"theta={obs[0]:.4f}  "
                f"theta_dot={obs[1]:.4f}  "
                f"wind={obs[2]:.4f}  "
                f"wave={obs[3]:.4f}  "
                f"reward={reward:.4f}  "
                f"term={term}  "
                f"trunc={trunc}"
            )
            assert obs.shape == (4,)
            assert isinstance(reward, float) or np.isscalar(reward)
            if term:
                print("  ⚠  terminated by safety threshold")
                break

        print("  ✓  passed")
        passed += 1

    # Test randomized_training mode
    print("\n" + "="*50)
    print("Testing randomized_training mode")
    print("="*50)

    env = FloatingPlatformEnv(randomized_training=True)
    obs_a, _ = env.reset(seed=1)
    stiffness_a = env.stiffness

    obs_b, _ = env.reset(seed=2)
    stiffness_b = env.stiffness

    assert abs(stiffness_a - stiffness_b) > 1e-3, (
        "randomized_training should produce different parameters per reset"
    )
    print(f"  Reset 1 stiffness={stiffness_a:.3f}")
    print(f"  Reset 2 stiffness={stiffness_b:.3f}")
    print("  ✓  passed")
    passed += 1

    # Test seed reproducibility
    print("\n" + "="*50)
    print("Testing seed reproducibility")
    print("="*50)

    env1 = FloatingPlatformEnv(scenario="normal_wind")
    env2 = FloatingPlatformEnv(scenario="normal_wind")

    obs1, _ = env1.reset(seed=123)
    obs2, _ = env2.reset(seed=123)

    np.testing.assert_allclose(obs1, obs2, atol=1e-6)
    print(f"  obs1={obs1.tolist()}")
    print(f"  obs2={obs2.tolist()}")
    print("  ✓  passed")
    passed += 1

    # Test info dict
    print("\n" + "="*50)
    print("Testing info dict contents")
    print("="*50)

    env = FloatingPlatformEnv(scenario="strong_wind")
    obs, _ = env.reset(seed=7)
    _, _, _, _, info = env.step(env.action_space.sample())

    for key in ("theta", "theta_dot", "wind", "wave", "action", "terminated_by_safety"):
        assert key in info, f"Missing key '{key}' in info dict"
    print(f"  info keys: {list(info.keys())}")
    print("  ✓  passed")
    passed += 1

    print(f"\n{'='*50}")
    print(f"All {passed} / {passed} tests passed.")
    print("="*50)
