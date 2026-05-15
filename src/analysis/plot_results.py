"""Plotting script for evaluation results and training logs.

Reads evaluation_summary.csv and trajectory CSVs from the metrics directory,
generates comparison figures, and optionally plots training learning curves.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DPI = 150

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame | None:
    """Load a CSV file, return None if missing."""
    if not path.exists():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Base (Round 1 / generic) plots
# ---------------------------------------------------------------------------

def load_metrics(metrics_dir: Path) -> pd.DataFrame | None:
    """Load evaluation_summary.csv if it exists."""
    summary_path = metrics_dir / "evaluation_summary.csv"
    if summary_path.exists():
        return pd.read_csv(summary_path)
    return None


def plot_pitch_angle_comparison(metrics_dir: Path, figures_dir: Path,
                                suffix: str = ""):
    """Plot theta over time for normal_wind trajectory."""
    base = f"trajectories_normal_wind{suffix}.csv"
    traj_path = metrics_dir / base
    if not traj_path.exists():
        print(f"  [skip] {traj_path} not found")
        return

    df = pd.read_csv(traj_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["theta"], label="Agent trajectory (first episode)", linewidth=0.8, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Pitch angle (rad)")
    ax.set_title(f"Platform Pitch Angle -- normal_wind scenario{(' '+suffix) if suffix else ''}")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(0.3, color="red", linewidth=0.8, linestyle="--", alpha=0.6, label="Safety threshold (0.3 rad)")
    ax.axhline(-0.3, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outname = f"pitch_angle_comparison{suffix}.png"
    path = figures_dir / outname
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_control_action_comparison(metrics_dir: Path, figures_dir: Path,
                                   suffix: str = ""):
    """Plot action over time for normal_wind trajectory."""
    base = f"trajectories_normal_wind{suffix}.csv"
    traj_path = metrics_dir / base
    if not traj_path.exists():
        print(f"  [skip] {traj_path} not found")
        return

    df = pd.read_csv(traj_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["action"], label="Control action", linewidth=0.8, color="steelblue")
    ax.set_xlabel("Step")
    ax.set_ylabel("Action (normalised)")
    ax.set_title(f"Control Action -- normal_wind scenario{(' '+suffix) if suffix else ''}")
    ax.axhline(1.0, color="red", linewidth=0.5, linestyle="--", alpha=0.7, label="Action limits")
    ax.axhline(-1.0, color="red", linewidth=0.5, linestyle="--", alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outname = f"control_action_comparison{suffix}.png"
    path = figures_dir / outname
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_wind_wave_disturbance(metrics_dir: Path, figures_dir: Path,
                               suffix: str = ""):
    """Plot wind and wave disturbances over time (normal_wind)."""
    base = f"trajectories_normal_wind{suffix}.csv"
    traj_path = metrics_dir / base
    if not traj_path.exists():
        print(f"  [skip] {traj_path} not found")
        return

    df = pd.read_csv(traj_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["wind"], label="Wind disturbance", linewidth=0.8, color="orange")
    ax.plot(df["wave"], label="Wave disturbance", linewidth=0.8, color="teal", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Disturbance (N*m)")
    ax.set_title(f"Wind & Wave Disturbances -- normal_wind scenario{(' '+suffix) if suffix else ''}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    outname = f"wind_wave_disturbance{suffix}.png"
    path = figures_dir / outname
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_failure_rate_bar(metrics_dir: Path, figures_dir: Path,
                          suffix: str = ""):
    """Bar chart of failure rate by controller x scenario."""
    fname = f"evaluation_summary{suffix}.csv"
    summary_path = metrics_dir / fname
    if not summary_path.exists():
        print(f"  [skip] {summary_path} not found")
        return

    df = pd.read_csv(summary_path)
    pivot = df.pivot_table(index="controller", columns="scenario", values="failure_rate")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Failure rate")
    ax.set_title(f"Failure rate by controller and scenario{(' '+suffix) if suffix else ''}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Scenario", loc="upper right")
    fig.tight_layout()
    outname = f"failure_rate_bar{suffix}.png"
    path = figures_dir / outname
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_robustness_comparison(metrics_dir: Path, figures_dir: Path,
                               suffix: str = ""):
    """Grouped bar chart showing avg_return by scenario for each controller."""
    fname = f"evaluation_summary{suffix}.csv"
    summary_path = metrics_dir / fname
    if not summary_path.exists():
        print(f"  [skip] {summary_path} not found")
        return

    df = pd.read_csv(summary_path)
    pivot = df.pivot_table(index="controller", columns="scenario", values="avg_return")

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.set_ylabel("Average return")
    ax.set_title(f"Robustness: Average return by controller and scenario{(' '+suffix) if suffix else ''}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Scenario", loc="best")
    fig.tight_layout()
    outname = f"robustness_comparison{suffix}.png"
    path = figures_dir / outname
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_learning_curve(logs_dir: Path, figures_dir: Path):
    """Plot training reward curve if a training log CSV exists."""
    log_path = logs_dir / "ppo_normal_wind_training_log.csv"
    if not log_path.exists():
        log_path = logs_dir / "training_log.csv"
    if not log_path.exists():
        log_path = logs_dir / "progress.csv"
    if not log_path.exists():
        # Fallback: look for any .monitor.csv from VecMonitor
        monitor_files = sorted(logs_dir.glob("*.monitor.csv"))
        if monitor_files:
            log_path = monitor_files[-1]  # most recent
    if not log_path.exists():
        print(f"  [skip] No training log CSV found in {logs_dir}")
        return

    try:
        df = pd.read_csv(log_path, comment="#")
    except Exception as e:
        print(f"  [skip] Could not parse {log_path}: {e}")
        return

    reward_col = None
    for candidate in ["r", "ep_rew_mean", "rollout/ep_rew_mean", "avg_return", "reward", "mean_reward"]:
        if candidate in df.columns:
            reward_col = candidate
            break

    if reward_col is None:
        print(f"  [skip] No reward column found. Available: {list(df.columns)}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    values = df[reward_col].values
    valid = np.isfinite(values)
    if not np.any(valid):
        print(f"  [skip] No valid values in {reward_col}")
        return

    x = np.arange(len(values))[valid]
    y = values[valid]

    ax.plot(x, y, linewidth=0.8, color="steelblue", alpha=0.6, label=reward_col)
    if len(y) > 20:
        window = max(20, len(y) // 10)
        cumsum = np.cumsum(np.insert(y, 0, 0))
        smoothed = (cumsum[window:] - cumsum[:-window]) / window
        ax.plot(
            range(window - 1, len(values)),
            smoothed,
            linewidth=2,
            color="darkred",
            alpha=0.8,
            label=f"Moving avg ({window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Training learning curve")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / "learning_curve.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Round 3 plots (error-bar / multi-seed)
# ---------------------------------------------------------------------------

# Expected 4 evaluation scenarios for the grouped charts.
# These are the scenario names used in the evaluation summaries.
# The script derives these dynamically if the aggregated CSV uses different names.
R3_DEFAULT_SCENARIOS = ["normal_wind", "strong_wind", "variable_wind",
                        "out_of_distribution_wind"]

# Display-friendly labels for scenarios
SCENARIO_LABELS = {
    "normal_wind": "Normal",
    "strong_wind": "Strong",
    "variable_wind": "Variable",
    "out_of_distribution_wind": "OOD",
}

# Groups / agents displayed on the x-axis for grouped bar charts.
R3_GROUPS = ["PD", "PPO", "PPO-Randomized", "PPO+Safety", "PPO-Randomized+Safety"]


def _get_scenarios(aggregated: pd.DataFrame) -> list[str]:
    """Return the scenario column names from the aggregated CSV."""
    candidates = ["scenario", "Scenario", "env_name", "environment"]
    for c in candidates:
        if c in aggregated.columns:
            return sorted(aggregated[c].unique().tolist())
    # Fallback: assume the index or first non-metric columns
    return R3_DEFAULT_SCENARIOS


def _label_scenario(s: str) -> str:
    """Return a display-friendly label for a scenario name."""
    return SCENARIO_LABELS.get(s, s.replace("_", " ").title())


def _grouped_bar_with_errorbars(
    aggregated: pd.DataFrame,
    scenarios: list[str],
    groups: list[str],
    value_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    fig_path: Path,
    figsize: tuple = (11, 5.5),
    y_min: float | None = None,
    y_max: float | None = None,
    rotate_xticklabels: int = 0,
):
    """Generic grouped bar chart with error bars.

    Parameters
    ----------
    aggregated : pd.DataFrame
        The *aggregated* CSV loaded as a DataFrame. Each row = one
        (group, scenario) pair with mean and std columns.
    scenarios : list[str]
        Ordered list of scenario values to plot (used as sub-groups).
    groups : list[str]
        Ordered list of group / controller values (used as x-axis ticks).
    value_col : str
        Column name holding the mean value.
    std_col : str
        Column name holding the standard deviation.
    ylabel, title, fig_path : str / Path
    """
    fig, ax = plt.subplots(figsize=figsize)

    n_scenarios = len(scenarios)
    x_groups = np.arange(len(groups))       # bar group centres
    bar_width = 0.8 / n_scenarios            # width of each individual bar

    palette = plt.get_cmap("tab10")

    for j, scenario in enumerate(scenarios):
        offset = (j - (n_scenarios - 1) / 2) * bar_width
        means = []
        stds = []
        for g in groups:
            # Support either 'model' or 'controller' column name
            model_col = "model" if "model" in aggregated.columns else "controller"
            mask = (aggregated[model_col] == g) & (aggregated["scenario"] == scenario)
            subset = aggregated[mask]
            if len(subset) == 1:
                means.append(subset.iloc[0][value_col])
                stds.append(subset.iloc[0][std_col])
            else:
                # If the aggregated CSV already has the right shape, take the row
                means.append(np.nan)
                stds.append(np.nan)
        # If means contains NaN, try to re-pull from raw aggregated
        if any(np.isnan(m) for m in means):
            means = []
            stds = []
            model_col = "model" if "model" in aggregated.columns else "controller"
            for g in groups:
                mask = (aggregated[model_col] == g) & (aggregated["scenario"] == scenario)
                subset = aggregated[mask]
                if len(subset) == 1:
                    means.append(float(subset.iloc[0][value_col]))
                    stds.append(float(subset.iloc[0][std_col]))
                else:
                    means.append(0.0)
                    stds.append(0.0)

        color = palette(j)
        ax.bar(
            x_groups + offset,
            means,
            bar_width,
            label=_label_scenario(scenario),
            color=color,
            edgecolor="black",
            linewidth=0.5,
            yerr=stds,
            capsize=4,
            error_kw={"elinewidth": 1.2},
        )

    ax.set_xticks(x_groups)
    ax.set_xticklabels(groups, rotation=rotate_xticklabels, ha="center")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if y_min is not None or y_max is not None:
        ymin = y_min if y_min is not None else ax.get_ylim()[0]
        ymax = y_max if y_max is not None else ax.get_ylim()[1]
        ax.set_ylim(ymin, ymax)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Scenario", loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {fig_path}")


def plot_round3_average_return_mean_std(metrics_dir: Path, figures_dir: Path,
                                        aggregated: pd.DataFrame):
    """Grouped bar chart of avg_return (mean +/- std) across scenarios."""
    scenarios = _get_scenarios_from_agg(aggregated)
    # Support both column naming conventions
    ret_col = "avg_return_mean" if "avg_return_mean" in aggregated.columns else "avg_return"
    ret_std = "avg_return_std"
    _grouped_bar_with_errorbars(
        aggregated, scenarios, R3_GROUPS,
        value_col=ret_col, std_col=ret_std,
        ylabel="Average Return (mean +/- std)",
        title="Round 3: Average Return by Controller and Scenario",
        fig_path=figures_dir / "round3_average_return_mean_std.png",
    )


def plot_round3_failure_rate_mean_std(metrics_dir: Path, figures_dir: Path,
                                      aggregated: pd.DataFrame):
    """Grouped bar chart of failure rate (mean +/- std)."""
    scenarios = _get_scenarios_from_agg(aggregated)
    fail_col = "failure_rate_mean" if "failure_rate_mean" in aggregated.columns else "failure_rate"
    fail_std = "failure_rate_std"
    _grouped_bar_with_errorbars(
        aggregated, scenarios, R3_GROUPS,
        value_col=fail_col, std_col=fail_std,
        ylabel="Failure Rate (mean +/- std)",
        title="Round 3: Failure Rate by Controller and Scenario",
        fig_path=figures_dir / "round3_failure_rate_mean_std.png",
        y_min=0.0, y_max=1.0,
    )


def plot_round3_mean_abs_pitch_mean_std(metrics_dir: Path, figures_dir: Path,
                                        aggregated: pd.DataFrame):
    """Grouped bar chart of mean_abs_theta (mean +/- std)."""
    scenarios = _get_scenarios_from_agg(aggregated)
    pitch_col = "mean_abs_theta_mean" if "mean_abs_theta_mean" in aggregated.columns else "mean_abs_theta"
    pitch_std_col = "mean_abs_theta_std"
    _grouped_bar_with_errorbars(
        aggregated, scenarios, R3_GROUPS,
        value_col=pitch_col, std_col=pitch_std_col,
        ylabel="Mean |Pitch Angle| (rad, mean +/- std)",
        title="Round 3: Mean |Pitch Angle| by Controller and Scenario",
        fig_path=figures_dir / "round3_mean_abs_pitch_mean_std.png",
        y_min=0.0,
    )


def plot_round3_control_energy_mean_std(metrics_dir: Path, figures_dir: Path,
                                        aggregated: pd.DataFrame):
    """Grouped bar chart of control energy (mean +/- std)."""
    scenarios = _get_scenarios_from_agg(aggregated)
    energy_col = "control_energy_mean" if "control_energy_mean" in aggregated.columns else "control_energy"
    energy_std_col = "control_energy_std"
    _grouped_bar_with_errorbars(
        aggregated, scenarios, R3_GROUPS,
        value_col=energy_col, std_col=energy_std_col,
        ylabel="Control Energy (mean +/- std)",
        title="Round 3: Control Energy by Controller and Scenario",
        fig_path=figures_dir / "round3_control_energy_mean_std.png",
        y_min=0.0,
    )


def _get_scenarios_from_agg(aggregated: pd.DataFrame) -> list[str]:
    """Return the ordered scenario list from aggregated round-3 data."""
    # If the aggregated data has a 'scenario' column, use those values
    if "scenario" in aggregated.columns:
        # Try to order them in a sensible way
        scens = aggregated["scenario"].unique().tolist()
        ordered = []
        for s in R3_DEFAULT_SCENARIOS:
            if s in scens:
                ordered.append(s)
        for s in scens:
            if s not in ordered:
                ordered.append(s)
        return ordered
    return R3_DEFAULT_SCENARIOS


def plot_round3_robustness_gap(metrics_dir: Path, figures_dir: Path,
                               aggregated: pd.DataFrame):
    """Bar chart showing the 'robustness gap':
    (OOD avg_return - normal_wind avg_return) for each controller,
    with std computed via error propagation."""
    scenarios = _get_scenarios_from_agg(aggregated)
    fig, ax = plt.subplots(figsize=(9, 5))

    # Identify normal and OOD scenario names
    normal_name = "normal_wind"
    ood_name = "out_of_distribution_wind"
    if normal_name not in scenarios or ood_name not in scenarios:
        # Try to find closest matches
        for s in scenarios:
            if "normal" in s and normal_name not in [normal_name]:
                normal_name = s
            if "out_of_distribution" in s or s == "ood" or s == "OOD":
                ood_name = s

    # Support either 'model' or 'controller' column name
    model_col = "model" if "model" in aggregated.columns else "controller"
    # Support both column naming conventions for avg_return
    ret_col = "avg_return_mean" if "avg_return_mean" in aggregated.columns else "avg_return"
    ret_std = "avg_return_std"
    gaps = []
    gap_stds = []
    for g in R3_GROUPS:
        mask_n = (aggregated[model_col] == g) & (aggregated["scenario"] == normal_name)
        mask_o = (aggregated[model_col] == g) & (aggregated["scenario"] == ood_name)
        row_n = aggregated[mask_n]
        row_o = aggregated[mask_o]
        if len(row_n) == 1 and len(row_o) == 1:
            mean_n = row_n.iloc[0][ret_col]
            std_n = row_n.iloc[0][ret_std]
            mean_o = row_o.iloc[0][ret_col]
            std_o = row_o.iloc[0][ret_std]
            gap = mean_o - mean_n
            # Propagate error: std(a-b) = sqrt(std_a^2 + std_b^2)
            gap_std = np.sqrt(std_o**2 + std_n**2)
            gaps.append(gap)
            gap_stds.append(gap_std)
        else:
            gaps.append(np.nan)
            gap_stds.append(np.nan)

    colors = plt.cm.tab10.colors
    labels_list = [_label_scenario(s) for s in scenarios]

    x = np.arange(len(R3_GROUPS))
    ax.bar(
        x, gaps, width=0.6, color=colors[:len(R3_GROUPS)],
        edgecolor="black", linewidth=0.5,
        yerr=gap_stds, capsize=5, error_kw={"elinewidth": 1.5},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(R3_GROUPS, ha="center")
    ax.set_ylabel(f"Return Gap: {ood_name} - {normal_name}")
    ax.set_title("Round 3: Robustness Gap (OOD - Normal) by Controller")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(figures_dir / "round3_robustness_gap.png", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {figures_dir / 'round3_robustness_gap.png'}")


def plot_round3_learning_curves_by_seed(logs_dir: Path, figures_dir: Path):
    """Plot per-seed learning curves for PPO-Normal and PPO-Randomized.

    Reads monitor CSVs of the form:
        ppo_normal_500k_seed{0,1,2}.monitor.csv
        ppo_randomized_500k_seed{0,1,2}.monitor.csv
    VecMonitor .monitor.csv files have a '#' comment header on line 1
    and columns r, l, t.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    styles = [
        ("ppo_normal", "PPO-Normal", ["steelblue", "dodgerblue", "cornflowerblue"]),
        ("ppo_randomized", "PPO-Randomized", ["darkorange", "orange", "gold"]),
    ]

    has_any = False
    for prefix, label, colors in styles:
        for seed_idx in range(3):
            monitor_path = logs_dir / f"{prefix}_500k_seed{seed_idx}.monitor.csv"
            if not monitor_path.exists():
                continue
            try:
                df = pd.read_csv(monitor_path, comment="#")
            except Exception:
                print(f"  [skip] Could not parse {monitor_path}")
                continue

            if "r" not in df.columns:
                print(f"  [skip] No 'r' column in {monitor_path}")
                continue

            rewards = df["r"].values
            valid = np.isfinite(rewards)
            if not np.any(valid):
                continue

            y = rewards[valid]
            x = np.cumsum(df["l"].values[valid]) if "l" in df.columns else np.arange(len(y))

            style = "-"
            ls = {"0": "-", "1": "--", "2": ":"}.get(str(seed_idx), "-")
            ax.plot(x, y, color=colors[seed_idx], linestyle=ls,
                    linewidth=0.9, alpha=0.7,
                    label=f"{label} (seed {seed_idx})")
            has_any = True

    if not has_any:
        print("  [skip] No Round 3 seed monitor files found for learning curves")
        plt.close(fig)
        return

    ax.set_xlabel("Cumulative Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title("Round 3: Learning Curves by Seed (PPO-Normal vs PPO-Randomized)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "round3_learning_curves_by_seed.png", dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {figures_dir / 'round3_learning_curves_by_seed.png'}")


def plot_round3_existing(metrics_dir: Path, figures_dir: Path):
    """Generate the existing plot types (pitch, control, failure_rate, robustness)
    for Round 3 trajectories / evaluation summary if available."""
    # Trajectory-based plots use _round3 suffix
    plot_pitch_angle_comparison(metrics_dir, figures_dir, suffix="_round3")
    plot_control_action_comparison(metrics_dir, figures_dir, suffix="_round3")
    plot_wind_wave_disturbance(metrics_dir, figures_dir, suffix="_round3")

    # Summary-based plots use evaluation_summary_round3.csv
    plot_failure_rate_bar(metrics_dir, figures_dir, suffix="_round3")
    plot_robustness_comparison(metrics_dir, figures_dir, suffix="_round3")


def run_round3_plots(metrics_dir: Path, figures_dir: Path, logs_dir: Path):
    """Execute all Round 3 specific plots."""
    print("\n=== Round 3: Aggregated Multi-Seed Plots ===")

    # ---- Load the aggregated Round 3 data ----
    agg_path = metrics_dir / "evaluation_summary_round3_aggregated.csv"
    if not agg_path.exists():
        # Try alternate name
        agg_path = metrics_dir / "evaluation_summary_round3.csv"

    if agg_path.exists():
        aggregated = pd.read_csv(agg_path)
        print(f"  Loaded: {agg_path}")

        # --- 6 new Round 3 plots ---
        plot_round3_average_return_mean_std(metrics_dir, figures_dir, aggregated)
        plot_round3_failure_rate_mean_std(metrics_dir, figures_dir, aggregated)
        plot_round3_mean_abs_pitch_mean_std(metrics_dir, figures_dir, aggregated)
        plot_round3_control_energy_mean_std(metrics_dir, figures_dir, aggregated)
        plot_round3_robustness_gap(metrics_dir, figures_dir, aggregated)

        print("  Saved: 5 aggregated bar charts with error bars")
    else:
        print(f"  [warn] Neither {metrics_dir / 'evaluation_summary_round3_aggregated.csv'} "
              f"nor {metrics_dir / 'evaluation_summary_round3.csv'} found.")
        print("  Skipping aggregated bar charts (run evaluation first).")

    # --- Learning curves from per-seed monitor logs ---
    plot_round3_learning_curves_by_seed(logs_dir, figures_dir)

    # --- Existing plot types for Round 3 data ---
    print("\n=== Round 3: Standard plots from trajectories/summary ===")
    plot_round3_existing(metrics_dir, figures_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation and training plots.")
    parser.add_argument("--metrics-dir", type=str, default="results/metrics/")
    parser.add_argument("--figures-dir", type=str, default="results/figures/")
    parser.add_argument("--logs-dir", type=str, default="results/logs/")
    parser.add_argument(
        "--round", type=int, default=None, choices=[1, 2, 3],
        help="Generate plots for a specific round only. "
             "Round 1 = base plots (default behaviour if omitted), "
             "Round 2 = _round2-suffixed data, "
             "Round 3 = aggregated multi-seed plots with error bars.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    figures_dir = Path(args.figures_dir)
    logs_dir = Path(args.logs_dir)

    figures_dir.mkdir(parents=True, exist_ok=True)

    if args.round == 3:
        # Round 3 only
        print("Generating Round 3 plots...")
        run_round3_plots(metrics_dir, figures_dir, logs_dir)
    elif args.round == 2:
        # Round 2 only: use _round2 suffixes
        print("Generating Round 2 plots...")
        plot_pitch_angle_comparison(metrics_dir, figures_dir, suffix="_round2")
        plot_control_action_comparison(metrics_dir, figures_dir, suffix="_round2")
        plot_wind_wave_disturbance(metrics_dir, figures_dir, suffix="_round2")
        plot_failure_rate_bar(metrics_dir, figures_dir, suffix="_round2")
        plot_robustness_comparison(metrics_dir, figures_dir, suffix="_round2")
    elif args.round == 1:
        # Base plots only
        print("Generating Round 1 (base) plots...")
        plot_pitch_angle_comparison(metrics_dir, figures_dir)
        plot_control_action_comparison(metrics_dir, figures_dir)
        plot_wind_wave_disturbance(metrics_dir, figures_dir)
        plot_failure_rate_bar(metrics_dir, figures_dir)
        plot_robustness_comparison(metrics_dir, figures_dir)
        plot_learning_curve(logs_dir, figures_dir)
    else:
        # Default: all rounds
        print("Generating all plots...")

        # Round 1 (base)
        print("\n=== Round 1 (base) ===")
        plot_pitch_angle_comparison(metrics_dir, figures_dir)
        plot_control_action_comparison(metrics_dir, figures_dir)
        plot_wind_wave_disturbance(metrics_dir, figures_dir)
        plot_failure_rate_bar(metrics_dir, figures_dir)
        plot_robustness_comparison(metrics_dir, figures_dir)
        plot_learning_curve(logs_dir, figures_dir)

        # Round 2
        print("\n=== Round 2 ===")
        run_r2 = plot_pitch_angle_comparison(metrics_dir, figures_dir, suffix="_round2")
        plot_control_action_comparison(metrics_dir, figures_dir, suffix="_round2")
        plot_wind_wave_disturbance(metrics_dir, figures_dir, suffix="_round2")
        plot_failure_rate_bar(metrics_dir, figures_dir, suffix="_round2")
        plot_robustness_comparison(metrics_dir, figures_dir, suffix="_round2")

        # Round 3
        run_round3_plots(metrics_dir, figures_dir, logs_dir)

    print(f"\nPlots saved to: {figures_dir}")


if __name__ == "__main__":
    main()
