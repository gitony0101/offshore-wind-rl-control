"""
Proportional-Derivative (PD) baseline for the Floating Platform pitch control environment.

A ``PDController`` applies classical proportional-derivative feedback on the
platform pitch angle and angular velocity.  This is a standard control-theory
baseline that does not require learning: the gains can be tuned by hand or
via systematic sweep, providing a meaningful upper-bound reference for simple
classical control.  Any well-trained RL policy should ideally match or exceed
PD performance.

Control law
-----------
    action = -Kp * theta - Kd * theta_dot

where:
  * Kp (proportional gain): produces a corrective action proportional to the
    angular displacement.  Drives the platform back toward zero pitch.
  * Kd (derivative gain): produces a corrective action proportional to the
    angular velocity.  Provides damping to suppress oscillations and
    overshoot.

The negative sign ensures the action opposes the displacement (stabilising
feedback).  The final action is clipped to [-1, 1] to respect the environment's
action-space bounds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Add the project root so this file can be run standalone.
# ---------------------------------------------------------------------------
_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def run_episode_with_controller(env, controller):
    """Run a single episode and collect trajectory data.

    Parameters
    ----------
    env : gymnasium.Env
        The floating platform environment.
    controller : object
        Any object with a ``predict(obs) -> action`` method.

    Returns
    -------
    total_reward : float
        Sum of step rewards over the episode.
    thetas : list[float]
        Pitch angle at every step (post-step observation).
    actions : list[float]
        Action issued at every step.
    terminated_info : dict
        ``terminated`` and ``truncated`` booleans from the final step.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    thetas = []
    actions = []
    terminated, truncated = False, False

    while not (terminated or truncated):
        action = controller.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        thetas.append(float(obs[0]))
        actions.append(float(action))

    return total_reward, thetas, actions, {
        "terminated": terminated,
        "truncated": truncated,
    }


class PDController:
    """Proportional-Derivative (PD) feedback controller.

    Applies a linear combination of pitch angle and angular velocity to
    produce a stabilising control action.  Based on classical control theory,
    PD feedback is among the simplest yet most effective linear controllers
    for second-order systems like the platform pitch dynamics.
    """

    def __init__(self, Kp: float = 5.0, Kd: float = 2.0):
        """Initialise the PD controller with proportional and derivative gains.

        Parameters
        ----------
        Kp : float
            Proportional gain.  Higher values produce stronger corrective
            torque proportional to the pitch-angle error.  Controls how
            aggressively the controller tries to return to upright.
        Kd : float
            Derivative gain.  Higher values produce stronger damping
            proportional to angular velocity.  Suppresses oscillations and
            reduces overshoot but can amplify sensor noise.
        """
        self.Kp = Kp
        self.Kd = Kd

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Compute the PD control action.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector ``[theta, theta_dot, wind, wave]``.
            Only ``theta`` (index 0) and ``theta_dot`` (index 1) are used.

        Returns
        -------
        action : np.ndarray of shape (1,)
            PD control action clipped to the valid [-1, 1] range.
        """
        theta = obs[0]
        theta_dot = obs[1]

        # PD control law: negative feedback on position and velocity.
        action = -self.Kp * theta - self.Kd * theta_dot

        return np.clip(np.array([action]), -1.0, 1.0)

    def evaluate(
        self, env, n_episodes: int = 20, seed: int | None = None
    ) -> dict:
        """Evaluate the PD controller over multiple episodes.

        Parameters
        ----------
        env : gymnasium.Env
            The floating platform environment instance.
        n_episodes : int
            Number of evaluation episodes.
        seed : int | None
            Optional random seed for reproducibility.

        Returns
        -------
        metrics : dict
            Dictionary with keys:
            - ``avg_return``: mean cumulative reward across episodes.
            - ``mean_abs_theta``: mean absolute pitch angle across all
              steps of all episodes.
            - ``max_abs_theta``: maximum absolute pitch angle ever observed.
            - ``failure_rate``: fraction of episodes terminated by a
              safety violation.
            - ``control_energy``: mean sum-of-squared actions per episode,
              a proxy for actuation effort.
        """
        returns = []
        all_thetas = []
        max_abs = 0.0
        failures = 0
        control_energies = []

        for ep in range(n_episodes):
            ep_seed = seed + ep if seed is not None else None
            env.reset(seed=ep_seed)

            total_reward, thetas, actions, done_info = run_episode_with_controller(
                env, self
            )

            returns.append(total_reward)
            all_thetas.extend(thetas)
            max_abs = max(max_abs, max((abs(t) for t in thetas), default=0.0))
            if done_info["terminated"]:
                failures += 1

            energy = sum(a * a for a in actions)
            control_energies.append(energy)

        metrics = {
            "avg_return": float(np.mean(returns)),
            "mean_abs_theta": float(np.mean(np.abs(all_thetas))) if all_thetas else 0.0,
            "max_abs_theta": float(max_abs),
            "failure_rate": float(failures / n_episodes) if n_episodes > 0 else 0.0,
            "control_energy": float(np.mean(control_energies)) if control_energies else 0.0,
        }
        return metrics


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from src.envs.floating_platform_env import FloatingPlatformEnv

    print("PDController Demonstration")
    print("=" * 55)

    # Default gains
    print("\n--- Default gains (Kp=5.0, Kd=2.0) ---")
    for scenario in ("normal_wind", "strong_wind", "out_of_distribution_wind"):
        env = FloatingPlatformEnv(scenario=scenario)
        controller = PDController()
        metrics = controller.evaluate(env, n_episodes=20, seed=42)

        print(f"\nScenario: {scenario}")
        print(f"  Avg return      : {metrics['avg_return']:+.2f}")
        print(f"  Mean |theta|    : {metrics['mean_abs_theta']:.4f} rad")
        print(f"  Max |theta|     : {metrics['max_abs_theta']:.4f} rad")
        print(f"  Failure rate    : {metrics['failure_rate']:.1%}")
        print(f"  Control energy  : {metrics['control_energy']:.4f}")

    # Sweep a few gain combinations under strong_wind
    print("\n" + "-" * 55)
    print("Gain sweep (strong_wind scenario):")
    print("-" * 55)
    print(f"{'Kp':>5}  {'Kd':>5}  {'Avg Return':>12}  {'Mean |theta|':>12}  {'Failure':>10}  {'Energy':>10}")

    env = FloatingPlatformEnv(scenario="strong_wind")
    for Kp in (2.0, 5.0, 10.0):
        for Kd in (1.0, 2.0, 5.0):
            controller = PDController(Kp=Kp, Kd=Kd)
            metrics = controller.evaluate(env, n_episodes=20, seed=42)

            print(
                f"{Kp:5.1f}  {Kd:5.1f}  "
                f"{metrics['avg_return']:>12.2f}  "
                f"{metrics['mean_abs_theta']:>12.4f}  "
                f"{metrics['failure_rate']:>9.1%}  "
                f"{metrics['control_energy']:>10.4f}"
            )

    print("\nDone.")
