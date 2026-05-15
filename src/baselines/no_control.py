"""
No-control baseline for the Floating Platform pitch control environment.

A ``NoControlController`` always outputs zero action, representing the natural
behaviour of the platform subject to wind and wave disturbances with no active
stabilisation.  This baseline serves as a lower-bound reference: any useful
RL or classical controller should outperform it in terms of pitch-angle
stabilisation and energy usage.
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


class NoControlController:
    """Baseline controller that always outputs zero action.

    This represents the uncontrolled dynamics of the floating platform —
    the pitch angle evolves solely under restoring stiffness, damping,
    and stochastic wind / wave disturbances.
    """

    def __init__(self) -> None:
        """Initialise the no-control baseline."""
        pass

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Return a zero action regardless of observation.

        Parameters
        ----------
        obs : np.ndarray
            Current observation ``[theta, theta_dot, wind, wave]``.

        Returns
        -------
        action : np.ndarray of shape (1,)
            Zero control effort, clipped to the valid action range.
        """
        action = np.clip(np.array([0.0]), -1.0, 1.0)
        return action

    def evaluate(
        self, env, n_episodes: int = 20, seed: int | None = None
    ) -> dict:
        """Evaluate the controller over multiple episodes.

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
            - ``control_energy``: mean sum-of-squared actions per episode
              (always zero for this controller).
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

    print("NoControlController Demonstration")
    print("=" * 55)

    for scenario in ("normal_wind", "strong_wind", "out_of_distribution_wind"):
        env = FloatingPlatformEnv(scenario=scenario)
        controller = NoControlController()
        metrics = controller.evaluate(env, n_episodes=20, seed=42)

        print(f"\nScenario: {scenario}")
        print(f"  Avg return      : {metrics['avg_return']:+.2f}")
        print(f"  Mean |theta|    : {metrics['mean_abs_theta']:.4f} rad")
        print(f"  Max |theta|     : {metrics['max_abs_theta']:.4f} rad")
        print(f"  Failure rate    : {metrics['failure_rate']:.1%}")
        print(f"  Control energy  : {metrics['control_energy']:.4f}")

    print("\nDone.")
