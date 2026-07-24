"""Tables, summaries, plots, and export for roundtrip experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULT_COLUMNS = [
    "experiment_id",
    "benchmark",
    "robot_type",
    "dof",
    "number_of_goals",
    "difficulty",
    "base_planner",
    "order_method",
    "run",
    "seed",
    "success",
    "planning_time",
    "final_path_length",
    "final_path_points",
    "successful_subpaths",
    "failed_subpaths",
    "roadmap_nodes",
    "roadmap_edges",
    "point_collision_checks",
    "line_collision_checks",
    "exact_line_collision_checks",
    "collision_checks",
    "error",
]


def results_dataframe(records):
    """Convert experiment records into a stable DataFrame schema."""
    rows = [record.metrics for record in records]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def summarize_results(dataframe):
    """Aggregate repeated runs for report tables."""
    if dataframe.empty:
        return pd.DataFrame()

    group_columns = [
        "robot_type",
        "benchmark",
        "number_of_goals",
        "difficulty",
        "base_planner",
        "order_method",
    ]
    return (
        dataframe.groupby(group_columns, dropna=False)
        .agg(
            runs=("success", "size"),
            success_rate=("success", "mean"),
            planning_time_mean=("planning_time", "mean"),
            planning_time_std=("planning_time", "std"),
            final_path_length_mean=("final_path_length", "mean"),
            final_path_points_mean=("final_path_points", "mean"),
            successful_subpaths_mean=("successful_subpaths", "mean"),
            failed_subpaths_mean=("failed_subpaths", "mean"),
            roadmap_nodes_mean=("roadmap_nodes", "mean"),
            roadmap_edges_mean=("roadmap_edges", "mean"),
            collision_checks_mean=("collision_checks", "mean"),
        )
        .reset_index()
    )


def plot_success_rate(dataframe):
    """Compare overall success rates by base planner and order method."""
    rates = (
        dataframe.groupby(["base_planner", "order_method"])["success"]
        .mean()
        .mul(100.0)
        .unstack("order_method")
    )
    ax = rates.plot(kind="bar", figsize=(9, 5), ylim=(0, 100))
    ax.set_ylabel("Success rate [%]")
    ax.set_title("Roundtrip success rate")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_planner_metric(
    dataframe,
    metric,
    ylabel,
    successful_only=False,
):
    """Compare one metric across base planners and order methods."""
    selected = dataframe
    if successful_only:
        selected = selected[selected["success"]]
    selected = selected.dropna(subset=[metric])
    if selected.empty:
        raise ValueError(f"No values are available for {metric!r}.")

    values = (
        selected.groupby(["base_planner", "order_method"])[metric]
        .mean()
        .unstack("order_method")
    )
    ax = values.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel(ylabel)
    ax.set_title(metric.replace("_", " ").title())
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_metric_by_benchmark(
    dataframe,
    metric,
    ylabel,
    *,
    order_method="exact",
    successful_only=False,
):
    """Compare base planners separately in every benchmark environment."""
    selected = dataframe[dataframe["order_method"] == order_method]
    if successful_only:
        selected = selected[selected["success"]]
    selected = selected.dropna(subset=[metric])
    if selected.empty:
        raise ValueError(f"No values are available for {metric!r}.")

    values = (
        selected.groupby(["benchmark", "base_planner"])[metric]
        .mean()
        .unstack("base_planner")
    )
    ax = values.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{metric.replace('_', ' ').title()} by benchmark "
        f"({order_method} order)"
    )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax


def plot_required_comparisons(dataframe):
    """Create figures for every comparison named in the project summary."""
    metrics = (
        ("planning_time", "planning_time", "Planning time [s]", False),
        ("tour_length", "final_path_length", "Final roundtrip length", True),
        ("path_points", "final_path_points", "Points in final path", True),
        (
            "successful_subpaths",
            "successful_subpaths",
            "Successful pairwise paths",
            False,
        ),
        (
            "failed_subpaths",
            "failed_subpaths",
            "Failed pairwise paths",
            False,
        ),
        ("roadmap_nodes", "roadmap_nodes", "Total roadmap nodes", False),
        ("roadmap_edges", "roadmap_edges", "Total roadmap edges", False),
        (
            "collision_checks",
            "collision_checks",
            "Collision checks",
            False,
        ),
    )

    figures = {"success_rate": plot_success_rate(dataframe)}
    for name, metric, ylabel, successful_only in metrics:
        try:
            figures[name] = plot_planner_metric(
                dataframe,
                metric,
                ylabel,
                successful_only,
            )
        except ValueError:
            figures[name] = None
    return figures


def save_results_csv(dataframe, path="results/roundtrip_results.csv"):
    """Save raw metrics without serializing planner objects."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return output_path
