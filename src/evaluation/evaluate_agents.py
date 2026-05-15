"""Evaluation script for comparing RL agents against baselines.

Runs trained agents and baseline controllers (NoControl, PD, PPO, 
PPO-Randomized, PPO+Safety, PPO-Randomized+Safety) on multiple test 
scenarios, collects performance metrics (including intervention_rate),
and saves results to CSV for downstream analysis and plotting.

Round 3 support: --round 3 triggers multi-seed evaluation across
6 PPO models (3 standard + 3 randomized, seeds 0,1,2) alongside
PD and NoControl baselines.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_project_root = str(Path(__file__).parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.baselines.no_control import NoControlController
from src.baselines.pd_controller import PDController
from src.envs.floating_platform_env import FloatingPlatformEnv

try:
    from stable_baselines3 import PPO
    from src.safety.simple_safety_filter import SafetyFilteredController
except ImportError:
    PPO = None
    SafetyFilteredController = None


# ---------------------------------------------------------------------------
# Episode & evaluation helpers
# ---------------------------------------------------------------------------

def run_episode_with_trajectory(env, controller, seed=None):
    """Run a single episode and collect full trajectory data."""
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    traj = []
    terminated, truncated = False, False

    # Track trajectory-level safety info if the controller exposes it
    intervention_count = 0
    total_steps = 0

    while not (terminated or truncated):
        action = controller.predict(obs)
        # SB3 PPO returns (action, hidden_state) for LSTM policies.
        # For MlpPolicy it returns just the action, but guard against tuples.
        if isinstance(action, tuple):
            action = action[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_steps += 1
        traj.append({
            "theta": float(next_obs[0]),
            "theta_dot": float(next_obs[1]),
            "wind": float(next_obs[2]),
            "wave": float(next_obs[3]),
            "action": float(np.asarray(action).item()),
            "reward": float(reward),
        })
        obs = next_obs

        # Track interventions from SafetyFilteredController
        if hasattr(controller, "intervention_count"):
            intervention_count = controller.intervention_count

    intervention_rate = float(intervention_count / total_steps) if total_steps > 0 else 0.0

    return total_reward, traj, {"terminated": terminated, "truncated": truncated}, intervention_rate


def evaluate_controller(env, controller, n_episodes=20, seed=None):
    """Evaluate a controller over multiple episodes."""
    returns = []
    all_thetas = []
    max_abs = 0.0
    failures = 0
    control_energies = []
    intervention_rates = []
    first_traj = None

    for ep in range(n_episodes):
        ep_seed = seed + ep if seed is not None else None
        result = run_episode_with_trajectory(
            env, controller, seed=ep_seed
        )
        total_reward, traj, done_info, int_rate = result

        if ep == 0:
            first_traj = traj

        returns.append(total_reward)
        all_thetas.extend([s["theta"] for s in traj])
        episode_max = max((abs(s["theta"]) for s in traj), default=0.0)
        max_abs = max(max_abs, episode_max)
        if done_info["terminated"]:
            failures += 1

        energy = sum(s["action"] ** 2 for s in traj)
        control_energies.append(energy)
        intervention_rates.append(int_rate)

    metrics = {
        "avg_return": float(np.mean(returns)),
        "mean_abs_theta": float(np.mean(np.abs(all_thetas))) if all_thetas else 0.0,
        "max_abs_theta": float(max_abs),
        "failure_rate": float(failures / n_episodes) if n_episodes > 0 else 0.0,
        "control_energy": float(np.mean(control_energies)) if control_energies else 0.0,
        "intervention_rate": float(np.mean(intervention_rates)) if any(r > 0 for r in intervention_rates) else 0.0,
        "n_episodes": n_episodes,
    }
    return metrics, first_traj


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

def _discover_ppo_models(model_dir: Path, round_num: int):
    """Return a dict of controller_name -> model_path for the requested round.

    Round 1/2 (default): single PPO and PPO-Randomized model with fallbacks.
    Round 3:           6 multi-seed models (3 standard + 3 randomized, seeds 0,1,2).
    """
    model_paths: dict[str, Optional[Path]] = {}

    if round_num == 3:
        # Round 3 multi-seed models
        for seed in (0, 1, 2):
            normal_path = model_dir / f"ppo_normal_500k_seed{seed}.zip"
            randomized_path = model_dir / f"ppo_randomized_500k_seed{seed}.zip"
            name_n = f"PPO-seed{seed}"
            name_r = f"PPO-Randomized-seed{seed}"
            model_paths[name_n] = normal_path if normal_path.exists() else None
            model_paths[name_r] = randomized_path if randomized_path.exists() else None
    else:
        # Original Round 1/2 behaviour
        model_paths["PPO"] = model_dir / "ppo_normal_50k.zip"
        model_paths["PPO-Randomized"] = model_dir / "ppo_randomized_50k.zip"

        # Fallbacks for legacy naming
        if not (model_paths["PPO"] or (model_dir / "ppo_normal_500k_seed0.zip").exists()):
            fb = model_dir / "ppo_normal_wind.zip"
            model_paths["PPO"] = fb if fb.exists() else None
        elif not (model_paths["PPO"] or (model_dir / "ppo_normal_50k.zip").exists()):
            fb = model_dir / "ppo_normal_500k_seed0.zip"
            model_paths["PPO"] = fb if fb.exists() else None

        if not (model_paths["PPO-Randomized"] or (model_dir / "ppo_randomized_500k_seed0.zip").exists()):
            fb = model_dir / "ppo_randomized_wind.zip"
            model_paths["PPO-Randomized"] = fb if fb.exists() else None
        elif not (model_paths["PPO-Randomized"] or (model_dir / "ppo_randomized_50k.zip").exists()):
            fb = model_dir / "ppo_randomized_500k_seed0.zip"
            model_paths["PPO-Randomized"] = fb if fb.exists() else None

    return model_paths


def _load_ppo_models(model_paths, ppo_filter_map=None):
    """Load PPO models and return (controllers dict, ppo_model_map).

    `ppo_filter_map` optional: dict of controller_name -> params
        e.g. {"PPO-seed0": {"safety_threshold": 0.3, ...}} to create
        safety-wrapped variants.
    """
    controllers: dict = {}
    ppo_model_map: dict = {}
    safety_params_default = {
        "safety_threshold": 0.3, "damping": 0.5,
        "stiffness": 1.5, "mass": 1.0, "dt": 0.05, "action_gain": 0.5,
    }

    for name, mpath in model_paths.items():
        if mpath is None or not mpath.exists():
            if PPO is not None:
                print(f"WARNING: PPO model NOT found ({name}) -- skipping.")
            continue

        if PPO is None:
            print(f"WARNING: stable_baselines3 not available, skipping {name}.")
            continue

        print(f"  Loading {name}: {mpath}")
        model = PPO.load(str(mpath))
        controllers[name] = model
        ppo_model_map[name] = model

        # Create safety-wrapped variant if applicable
        if ppo_filter_map is None or name in ppo_filter_map:
            safety_name = f"{name}+Safety"
            params = safety_params_default.copy()
            if ppo_filter_map and name in ppo_filter_map:
                params.update(ppo_filter_map[name])
            if SafetyFilteredController is not None:
                controllers[safety_name] = SafetyFilteredController(model, **params)

    return controllers, ppo_model_map


# ---------------------------------------------------------------------------
# Aggregation helper (Round 3)
# ---------------------------------------------------------------------------

def _aggregate_round3(rows_all):
    """Given per-run rows with 'seed' column, return aggregated rows (mean/std/min/max/failures) per controller family × scenario.

    Controller families: PPO (across seeds 0,1,2), PPO-Randomized, PD, NoControl.
    """
    from collections import defaultdict

    groups: dict[tuple, list] = defaultdict(list)  # (base_name, scenario) -> list of rows

    for row in rows_all:
        # Extract base controller name (strip seed suffix and +Safety suffix)
        ctrl = row.get("controller", "")
        base_name = ctrl
        # Strip +Safety suffix first if present (to handle e.g. "PPO-seed0+Safety")
        has_safety = base_name.endswith("+Safety")
        safety_tag = "+Safety" if has_safety else ""
        if has_safety:
            base_name = base_name[: -len("+Safety")]
        # Then strip seed suffix
        for suffix in ["-seed0", "-seed1", "-seed2"]:
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
                break
        # Re-apply safety tag so families are "PPO+Safety", "PPO-Randomized+Safety" etc.
        if safety_tag:
            base_name = base_name + safety_tag
        scenario = row.get("scenario", "")
        groups[(base_name, scenario)].append(row)

    metric_keys = ["avg_return", "mean_abs_theta", "max_abs_theta",
                   "failure_rate", "control_energy", "intervention_rate",
                   "n_episodes"]

    agg_rows = []
    for (base_name, scenario), group_rows in sorted(groups.items()):
        n = len(group_rows)
        agg = {"controller": base_name, "scenario": scenario}
        for key in metric_keys:
            values = [float(r.get(key, 0)) for r in group_rows]
            agg[f"{key}"] = float(np.mean(values))
            if n > 1:
                agg[f"{key}_std"] = float(np.std(values))
                agg[f"{key}_min"] = float(np.min(values))
                agg[f"{key}_max"] = float(np.max(values))
            else:
                agg[f"{key}_std"] = 0.0
                agg[f"{key}_min"] = float(values[0])
                agg[f"{key}_max"] = float(values[0])
        agg["n_seeds"] = n
        agg_rows.append(agg)

    return agg_rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RL agents and baselines across wind/wave scenarios."
    )
    parser.add_argument("--round", type=int, default=2,
                        help="Evaluation round. Use 3 for multi-seed evaluation.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", type=str, default="results/models/ppo_normal_wind.zip")
    parser.add_argument(
        "--scenarios",
        type=str,
        default="normal_wind,strong_wind,variable_wind,out_of_distribution_wind",
    )
    parser.add_argument("--output-dir", type=str, default="results/")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",")]
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "models"

    if args.round == 3:
        return _run_round3(args, scenarios, output_dir, metrics_dir, model_dir)
    else:
        return _run_round2(args, scenarios, output_dir, metrics_dir, model_dir)


def _run_round2(args, scenarios, output_dir, metrics_dir, model_dir):
    """Original Round 2 evaluation path."""
    model_paths = _discover_ppo_models(model_dir, round_num=2)

    # Apply legacy fallbacks the original script used
    ppo_path_v2 = model_paths.get("PPO")
    if ppo_path_v2 is None or not ppo_path_v2.exists():
        for fb_name in ("ppo_normal_wind.zip", "ppo_normal_500k_seed0.zip"):
            fb = model_dir / fb_name
            if fb.exists():
                model_paths["PPO"] = fb
                print(f"[INFO] Using fallback PPO model: {fb}")
                break
        else:
            model_paths["PPO"] = None

    ppo_rand_v2 = model_paths.get("PPO-Randomized")
    if ppo_rand_v2 is None or not ppo_rand_v2.exists():
        for fb_name in ("ppo_randomized_wind.zip", "ppo_randomized_500k_seed0.zip"):
            fb = model_dir / fb_name
            if fb.exists():
                model_paths["PPO-Randomized"] = fb
                print(f"[INFO] Using fallback PPO-Randomized model: {fb}")
                break
        else:
            model_paths["PPO-Randomized"] = None

    controllers = {
        "NoControl": NoControlController(),
        "PD": PDController(Kp=5.0, Kd=2.0),
    }

    if PPO is not None:
        _sc = SafetyFilteredController
    else:
        _sc = None

    for name in ("PPO", "PPO-Randomized"):
        mpath = model_paths.get(name)
        if mpath is None or not mpath.exists():
            if PPO is not None:
                print(f"WARNING: {name} model NOT found -- skipping.")
            continue
        if PPO is None:
            print(f"WARNING: stable_baselines3 not available, skipping {name}.")
            continue

        print(f"  Loading {name}: {mpath}")
        model = PPO.load(str(mpath))
        controllers[name] = model
        if _sc is not None and name == "PPO":
            controllers["PPO+Safety"] = _sc(
                model, safety_threshold=0.3, damping=0.5,
                stiffness=1.5, mass=1.0, dt=0.05, action_gain=0.5,
            )
        elif _sc is not None and name == "PPO-Randomized":
            controllers["PPO-Randomized+Safety"] = _sc(
                model, safety_threshold=0.3, damping=0.5,
                stiffness=1.5, mass=1.0, dt=0.05, action_gain=0.5,
            )

    all_rows = []
    trajectory_store = {}

    available_names = list(controllers.keys())
    print(f"\n{'='*70}")
    print(f"  Evaluation: {len(available_names)} controllers x {len(scenarios)} scenarios x {args.episodes} episodes")
    print(f"  Controllers: {', '.join(available_names)}")
    print(f"{'='*70}\n")

    for scenario in scenarios:
        env = FloatingPlatformEnv(scenario=scenario)
        print(f"--- Scenario: {scenario} ---")

        for name in available_names:
            ctrl = controllers[name]
            if hasattr(ctrl, "reset_intervention_count"):
                ctrl.reset_intervention_count()

            metrics, first_traj = evaluate_controller(
                env, ctrl, n_episodes=args.episodes, seed=args.seed
            )
            metrics["controller"] = name
            metrics["scenario"] = scenario
            all_rows.append(metrics)
            print(
                f"  {name:22s} | return={metrics['avg_return']:+.2f} | "
                f"mean|theta|={metrics['mean_abs_theta']:.4f} | "
                f"max|theta|={metrics['max_abs_theta']:.4f} | "
                f"fail={metrics['failure_rate']:.1%} | "
                f"intervene={metrics['intervention_rate']:.1%}"
            )
            trajectory_store[name] = first_traj

    # Save Round 2 CSVs
    _save_csv(all_rows, metrics_dir / "evaluation_summary_round2.csv")
    _save_csv(all_rows, metrics_dir / "evaluation_summary.csv")
    _save_trajectories(trajectory_store, metrics_dir, suffix="round2")
    _print_summary_table(all_rows)


def _run_round3(args, scenarios, output_dir, metrics_dir, model_dir):
    """Round 3 multi-seed evaluation.

    Discovers 6 PPO models (normal seeds 0,1,2  + randomized seeds 0,1,2),
    evaluates alongside PD baseline across all scenarios.
    Results include 'seed' column for aggregated analysis.
    """
    if PPO is None:
        print("ERROR: stable_baselines3 is required for Round 3 evaluation.")
        sys.exit(1)

    model_paths = _discover_ppo_models(model_dir, round_num=3)

    # Build controller registry: start with baselines
    controllers: dict = {}
    controllers["NoControl"] = NoControlController()
    controllers["PD"] = PDController(Kp=5.0, Kd=2.0)

    # We need to re-create the env per controller-seed-scenario combo
    # Actually: baselines have no model to load, we eval them once.
    # PPO models have per-seed controllers.

    # Separate baseline names from model-based names
    baseline_names = ["NoControl", "PD"]
    model_names = [n for n in model_paths if model_paths[n] is not None and model_paths[n].exists()]
    model_names.sort()

    # Also build safety-wrapped variants for the PPO models
    for name, mpath in model_paths.items():
        if mpath is None or not mpath.exists():
            print(f"WARNING: Model file not found for {name} -- skipping.")
            continue
        print(f"  Loading {name}: {mpath}")
        model_obj = PPO.load(str(mpath))
        controllers[name] = model_obj
        # Safety-wrapped variant
        safety_name = f"{name}+Safety"
        controllers[safety_name] = SafetyFilteredController(
            model_obj, safety_threshold=0.3, damping=0.5,
            stiffness=1.5, mass=1.0, dt=0.05, action_gain=0.5,
        )

    # Determine which names get evaluated
    # Baselines evaluated once; each PPO model variant evaluated once
    available_names = baseline_names + list(model_names)
    # Also include safety versions
    available_names += [n + "+Safety" for n in model_names if controllers.get(n) is not None]

    print(f"\n{'='*70}")
    print(f"  Round 3 Evaluation: {len(available_names)} controllers x {len(scenarios)} scenarios x {args.episodes} episodes")
    print(f"  Controllers: {', '.join(available_names)}")
    print(f"{'='*70}\n")

    all_rows = []
    trajectory_store = {}

    for scenario in scenarios:
        env = FloatingPlatformEnv(scenario=scenario)
        print(f"--- Scenario: {scenario} ---")

        for name in available_names:
            ctrl = controllers[name]
            if hasattr(ctrl, "reset_intervention_count"):
                ctrl.reset_intervention_count()

            # Determine seed: baselines use args.seed, PPO models use seed from name
            seed = args.seed
            if "-seed" in name:
                seed = int(name.split("-seed")[1].split("+")[0])

            metrics, first_traj = evaluate_controller(
                env, ctrl, n_episodes=args.episodes, seed=seed
            )
            metrics["controller"] = name
            metrics["scenario"] = scenario
            metrics["seed"] = seed
            all_rows.append(metrics)
            print(
                f"  {name:28s} | return={metrics['avg_return']:+.2f} | "
                f"mean|theta|={metrics['mean_abs_theta']:.4f} | "
                f"max|theta|={metrics['max_abs_theta']:.4f} | "
                f"fail={metrics['failure_rate']:.1%} | "
                f"intervene={metrics['intervention_rate']:.1%}"
            )
            trajectory_store[f"{scenario}_{name}"] = first_traj

    # ------------------------------------------------------------------
    # Save per-run Round 3 CSV
    # ------------------------------------------------------------------
    summary_path_r3 = metrics_dir / "evaluation_summary_round3.csv"
    fieldnames_r3 = [
        "controller", "scenario", "seed",
        "avg_return", "mean_abs_theta",
        "max_abs_theta", "failure_rate", "control_energy",
        "intervention_rate", "n_episodes",
    ]
    with open(summary_path_r3, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_r3)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row[k] for k in fieldnames_r3})
    print(f"\nPer-run metrics saved to: {summary_path_r3}")

    # ------------------------------------------------------------------
    # Save aggregated Round 3 CSV (mean/std/min/max per controller family × scenario)
    # ------------------------------------------------------------------
    agg_rows = _aggregate_round3(all_rows)
    summary_path_r3_agg = metrics_dir / "evaluation_summary_round3_aggregated.csv"
    agg_fieldnames = [
        "controller", "scenario", "n_seeds",
        "avg_return", "avg_return_std", "avg_return_min", "avg_return_max",
        "mean_abs_theta", "mean_abs_theta_std", "mean_abs_theta_min", "mean_abs_theta_max",
        "max_abs_theta", "max_abs_theta_std", "max_abs_theta_min", "max_abs_theta_max",
        "failure_rate", "failure_rate_std", "failure_rate_min", "failure_rate_max",
        "control_energy", "control_energy_std", "control_energy_min", "control_energy_max",
        "intervention_rate", "intervention_rate_std", "intervention_rate_min", "intervention_rate_max",
        "n_episodes",
    ]
    with open(summary_path_r3_agg, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fieldnames)
        writer.writeheader()
        for row in agg_rows:
            writer.writerow({k: row[k] for k in agg_fieldnames})
    print(f"Aggregated metrics saved to: {summary_path_r3_agg}")

    # ------------------------------------------------------------------
    # Save trajectory CSVs
    # ------------------------------------------------------------------
    for label, traj in trajectory_store.items():
        safe_label = label.replace(" ", "_")
        traj_path = metrics_dir / f"trajectories_{safe_label}_round3.csv"
        if traj is not None:
            t_fieldnames = ["step", "theta", "theta_dot", "wind", "wave", "action", "reward"]
            with open(traj_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=t_fieldnames)
                writer.writeheader()
                for i, step_data in enumerate(traj):
                    row = {"step": i}
                    row.update(step_data)
                    writer.writerow(row)
            print(f"Trajectory saved: {traj_path}")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    hdr = f"{'Controller':<28s} {'Scenario':<30s} {'Return':>10s} {'Mean|T|':>10s} {'Fail%':>8s} {'Energy':>10s} {'Interv%':>8s}"
    print(hdr)
    print(f"{'-'*70}")
    for row in all_rows:
        print(
            f"{row['controller']:<28s} {row['scenario']:<30s} "
            f"{row['avg_return']:>10.2f} {row['mean_abs_theta']:>10.4f} "
            f"{row['failure_rate']*100:>7.1f}% {row['control_energy']:>10.4f} "
            f"{row['intervention_rate']*100:>7.1f}%"
        )
    print(f"{'='*70}")


def _save_csv(rows, filepath):
    """Generic CSV saver using the Round 2 fieldnames."""
    fieldnames = [
        "controller", "scenario", "avg_return", "mean_abs_theta",
        "max_abs_theta", "failure_rate", "control_energy",
        "intervention_rate", "n_episodes",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})
    print(f"Metrics saved to: {filepath}")


def _save_trajectories(traj_store, metrics_dir, suffix=""):
    """Save trajectory CSVs."""
    suffix_label = f"_{suffix}" if suffix else ""
    for label, traj in traj_store.items():
        safe_label = label.replace(" ", "_")
        traj_path = metrics_dir / f"trajectories_{safe_label}{suffix_label}.csv"
        if traj is not None:
            t_fieldnames = ["step", "theta", "theta_dot", "wind", "wave", "action", "reward"]
            with open(traj_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=t_fieldnames)
                writer.writeheader()
                for i, step_data in enumerate(traj):
                    row = {"step": i}
                    row.update(step_data)
                    writer.writerow(row)
            print(f"Trajectory saved: {traj_path}")


def _print_summary_table(rows):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    hdr = f"{'Controller':<22s} {'Scenario':<30s} {'Return':>10s} {'Mean|T|':>10s} {'Fail%':>8s} {'Energy':>10s} {'Interv%':>8s}"
    print(hdr)
    print(f"{'-'*70}")
    for row in rows:
        print(
            f"{row['controller']:<22s} {row['scenario']:<30s} "
            f"{row['avg_return']:>10.2f} {row['mean_abs_theta']:>10.4f} "
            f"{row['failure_rate']*100:>7.1f}% {row['control_energy']:>10.4f} "
            f"{row['intervention_rate']*100:>7.1f}%"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
