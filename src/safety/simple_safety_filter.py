"""Simple safety filter for RL agent actions.

The safety filter performs a one-step-ahead prediction of the platform
pitch dynamics to decide whether a proposed control action would drive
the system dangerously close to (or beyond) the safety threshold.

Safety logic
------------
Given the current state (theta, theta_dot, wind, wave) and a proposed
action from any controller:

1. Compute the control force that would result from this action.
2. Integrate the equations of motion forward by one timestep (Euler
   method, using the same mass-spring-damper model as the environment).
3. If the predicted pitch angle exceeds the safety threshold, issue an
   emergency stop (action forced to zero).
4. If the predicted pitch angle exceeds a pre-warning band (80% of the
   threshold), scale down the action proportionally based on how deep
   the prediction is into the pre-warning zone.  The scale factor is
   ``1 - danger_ratio`` where::

       danger_ratio = (|theta_next| - 0.8*threshold) / (0.2*threshold)

   This ramps smoothly from 1.0 (no correction) at the pre-warning
   boundary to 0.0 (full stop) at the emergency boundary.
5. Otherwise, pass the original action through unchanged.

This provides a lightweight, model-based safeguard that requires no
training and works with any controller (PD, PPO, or custom).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np


def _apply_pre_warning_scale(
    safety_threshold: float,
    proposed_action: np.ndarray,
    abs_theta_next: float,
) -> np.ndarray:
    """Scale down an action proportionally within the pre-warning zone.

    Parameters
    ----------
    safety_threshold : float
        The safety threshold (radians).
    proposed_action : np.ndarray of shape (1,)
        The controller's proposed action.
    abs_theta_next : float
        Absolute value of the predicted pitch angle (>= 0).

    Returns
    -------
    np.ndarray of shape (1,)
        Scaled action clipped to [-1, 1].
    """
    pre_warning = safety_threshold * 0.8
    danger_ratio = (abs_theta_next - pre_warning) / (safety_threshold - pre_warning)
    danger_ratio = np.clip(danger_ratio, 0.0, 1.0)
    scale = 1.0 - danger_ratio
    return np.clip(proposed_action * scale, -1.0, 1.0).astype(np.float32)


def compute_intervention_rate(interventions_count: int, total_steps: int) -> float:
    """Compute the fraction of steps where safety intervention occurred.

    Parameters
    ----------
    interventions_count : int
        Number of safety filter interventions (from a
        ``SafetyFilteredController.intervention_count`` or equivalent).
    total_steps : int
        Total number of steps evaluated.

    Returns
    -------
    float
        Intervention rate in [0, 1], or 0.0 if total_steps is zero or
        negative.
    """
    if total_steps <= 0:
        return 0.0
    return interventions_count / total_steps


def safety_filter(
    state: np.ndarray,
    proposed_action: np.ndarray,
    safety_threshold: float = 0.3,
    damping: float = 0.5,
    stiffness: float = 1.5,
    mass: float = 1.0,
    dt: float = 0.05,
    action_gain: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply a one-step safety filter to a proposed control action.

    The filter predicts the one-step-ahead platform pitch angle given the
    proposed action and a mass-spring-damper model of the platform
    dynamics.  If the predicted pitch would exceed the safety threshold
    or enter a pre-warning zone, the action is modified accordingly.

    Parameters
    ----------
    state : np.ndarray of shape (4,)
        Current observation [theta, theta_dot, wind, wave].
    proposed_action : np.ndarray of shape (1,)
        The controller's proposed action (before safety intervention).
    safety_threshold : float, default 0.3
        Absolute pitch angle limit (radians).  0.3 rad ~ 17 degrees.
    damping : float, default 0.5
        Viscous damping coefficient (N·m·s/rad).
    stiffness : float, default 1.5
        Restoring torque coefficient (N·m/rad).
    mass : float, default 1.0
        Generalised mass (kg) of the pitch degree of freedom.
    dt : float, default 0.05
        Simulation timestep (seconds).
    action_gain : float, default 0.5
        Maps normalized action in [-1, 1] to control force (N·m).

    Returns
    -------
    original_action : np.ndarray of shape (1,)
        The proposed action, unchanged (returned for logging/analysis).
    filtered_action : np.ndarray of shape (1,)
        The action after safety processing (possibly modified or zeroed).
    was_intervened : bool
        True if the safety filter modified or disabled the action.

    Notes
    -----
    The prediction uses the same Euler-integration dynamics as
    FloatingPlatformEnv.step(), so the predicted state is consistent with
    what the environment would observe after taking the action.

    Interventions are graded:
    - Pre-warning (80% of threshold): action is scaled proportionally
      by (1 - danger_ratio) and clamped to [-1, 1], was_intervened=True.
    - Emergency stop (at or above threshold): action is set to 0.0,
      was_intervened=True.
    - Nominal: no change, was_intervened=False.
    """
    # Extract state components
    theta = float(state[0])
    theta_dot = float(state[1])
    wind = float(state[2])
    wave = float(state[3])

    # Compute control force from proposed action
    control_force = float(proposed_action[0]) * action_gain

    # Predict one step ahead using explicit Euler integration
    # (matches FloatingPlatformEnv.step() dynamics exactly)
    theta_next = theta + theta_dot * dt
    theta_dot_next = theta_dot + dt * (
        wind + wave + control_force - damping * theta_dot - stiffness * theta
    ) / mass

    original_action = proposed_action.copy()

    # Check against safety threshold — emergency stop takes precedence
    if abs(theta_next) > safety_threshold:
        # Emergency stop: zero out the action to prevent catastrophe
        filtered_action = np.array([0.0], dtype=np.float32)
        was_intervened = True

    elif abs(theta_next) > safety_threshold * 0.8:
        # Pre-warning: proportional scaling based on danger depth
        filtered_action = _apply_pre_warning_scale(
            safety_threshold=safety_threshold,
            proposed_action=proposed_action,
            abs_theta_next=abs(theta_next),
        )
        was_intervened = True

    else:
        # Nominal: pass through unchanged
        filtered_action = proposed_action.copy()
        was_intervened = False

    return original_action, filtered_action, was_intervened


class SafetyFilteredController:
    """Wrapper that applies safety filtering to any controller.

    This class wraps an arbitrary controller object that has a
    predict(obs) -> action method (e.g. a loaded SB3 policy, a PD
    controller, or a NoControl controller).  Every time predict or
    predict_safe is called, the proposed action is first generated by
    the wrapped controller, then passed through safety_filter before
    being returned.

    Parameters
    ----------
    controller : object
        Any object with a predict(obs) -> np.ndarray method.
    safety_threshold : float, default 0.3
        See safety_filter.
    damping : float, default 0.5
        See safety_filter.
    stiffness : float, default 1.5
        See safety_filter.
    mass : float, default 1.0
        See safety_filter.
    dt : float, default 0.05
        See safety_filter.
    action_gain : float, default 0.5
        See safety_filter.

    Attributes
    ----------
    intervention_count : int
        Number of times the safety filter intervened (modified or zeroed
        the action).  Reset to zero on reset_intervention_count.

    Examples
    --------
    >>> from stable_baselines3 import PPO
    >>> ppo = PPO.load("results/models/ppo_normal_wind.zip")
    >>> safe_wrapper = SafetyFilteredController(ppo)
    >>> safe_action = safe_wrapper.predict_safe(obs)
    """

    def __init__(
        self,
        controller: Any,
        safety_threshold: float = 0.3,
        damping: float = 0.5,
        stiffness: float = 1.5,
        mass: float = 1.0,
        dt: float = 0.05,
        action_gain: float = 0.5,
    ):
        self.controller = controller
        self.safety_threshold = safety_threshold
        self.damping = damping
        self.stiffness = stiffness
        self.mass = mass
        self.dt = dt
        self.action_gain = action_gain
        self.intervention_count = 0

    def predict_safe(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
        """Get filtered (safe) action plus intervention info.

        Parameters
        ----------
        obs : np.ndarray of shape (4,)
            Current observation [theta, theta_dot, wind, wave].

        Returns
        -------
        original_action : np.ndarray of shape (1,)
            The raw action from the wrapped controller.
        filtered_action : np.ndarray of shape (1,)
            The action after safety filtering.
        was_intervened : bool
            True if the safety filter modified the action.
        """
        proposed = self.controller.predict(obs)
        # SB3 PPO.predict() returns (action_array, state_dict) for LSTM policies.
        # Unwrap to get the raw numpy action array.
        if isinstance(proposed, tuple):
            proposed = proposed[0]
        # Normalise to a 1-d numpy array so proposed_action[0] is always safe
        if isinstance(proposed, (int, float)):
            proposed = np.array([float(proposed)], dtype=np.float32)
        else:
            proposed = np.asarray(proposed, dtype=np.float32).flatten()
            if proposed.ndim == 0:
                proposed = np.array([float(proposed)], dtype=np.float32)
        original, filtered, intervened = safety_filter(
            state=obs,
            proposed_action=proposed,
            safety_threshold=self.safety_threshold,
            damping=self.damping,
            stiffness=self.stiffness,
            mass=self.mass,
            dt=self.dt,
            action_gain=self.action_gain,
        )
        if intervened:
            self.intervention_count += 1
        return original, filtered, intervened

    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Get only the filtered (safe) action for API compatibility.

        This signature matches the predict method used by controllers in
        the NoControlController and PDController classes, as well as SB3
        policies.

        Parameters
        ----------
        obs : np.ndarray of shape (4,)
            Current observation.

        Returns
        -------
        filtered_action : np.ndarray of shape (1,)
            Safety-filtered action ready to be passed to env.step().
        """
        _, filtered, _ = self.predict_safe(obs)
        return filtered

    def reset_intervention_count(self) -> None:
        """Reset the intervention counter to zero."""
        self.intervention_count = 0
