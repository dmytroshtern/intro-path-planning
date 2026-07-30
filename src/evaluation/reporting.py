"""Plots for the PointRobot and PlanarManipulator evaluations."""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PLANNERS = ("BasicPRM", "VisibilityPRM", "LazyPRM")
_PLANNER_LABELS = _PLANNERS
_COMPARISON_PLANNERS = _PLANNERS + ("MultiQueryPRM",)
_ORDER_METHODS = ("exact", "greedy")

_BLUE = "#0000ff"
_YELLOW = "#d6cf00"
_PURPLE = "#800080"
_GREEN = "#2ca02c"
_RED = "#d62728"
_POINT_HEATMAP_COLORS = LinearSegmentedColormap.from_list(
    "point_evaluation_red_green",
    plt.colormaps["RdYlGn"](np.linspace(0.15, 0.85, 256)),
)


def _normalize_order_methods(order_methods):
    if isinstance(order_methods, str):
        order_methods = (order_methods,)
    order_methods = tuple(order_methods)
    unknown = set(order_methods) - set(_ORDER_METHODS)
    if unknown:
        raise ValueError(
            "Unknown order methods: " + ", ".join(sorted(unknown))
        )
    return order_methods


def _select_results(dataframe, benchmark_names, order_methods=None):
    if isinstance(benchmark_names, str):
        benchmark_names = (benchmark_names,)

    selected = dataframe[dataframe["benchmark"].isin(benchmark_names)].copy()
    if order_methods is not None:
        selected = selected[
            selected["order_method"].isin(order_methods)
        ]
    if selected.empty:
        raise ValueError("No matching benchmark results are available.")

    selected["successful_path_length"] = selected[
        "final_path_length"
    ].where(selected["success"])
    selected["successful_path_points"] = selected[
        "final_path_points"
    ].where(selected["success"])
    selected["roadmap_size"] = (
        selected["roadmap_nodes"] + selected["roadmap_edges"]
    )
    return selected


def _style_axis(axis, label, color):
    axis.set_ylabel(label, color=color)
    axis.tick_params(axis="y", labelcolor=color)
    axis.spines["left"].set_color(color)
    axis.grid(axis="y", alpha=0.25)


def _style_right_axis(axis, label, color, offset=None):
    if offset is not None:
        axis.spines["right"].set_position(("outward", offset))
    axis.set_ylabel(label, color=color)
    axis.tick_params(axis="y", labelcolor=color)
    axis.spines["right"].set_color(color)


def _metric_handles(metrics):
    return [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in metrics
    ]


def _set_planner_axis(axis, positions):
    axis.set_xticks(positions, _COMPARISON_PLANNERS)
    axis.set_xlabel("Base planner")


def _add_value_labels(axis, bars, value_format):
    labels = [
        value_format.format(bar.get_height())
        if np.isfinite(bar.get_height())
        else "n/a"
        for bar in bars
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)


def _add_segment_labels(axis, bars):
    labels = [
        f"{bar.get_height():.1f}" if bar.get_height() > 0 else ""
        for bar in bars
    ]
    axis.bar_label(
        bars,
        labels=labels,
        label_type="center",
        fontsize=8,
    )


def _heatmap(
    axis,
    values,
    row_labels,
    column_labels,
    *,
    title,
    color_map,
    colorbar_label,
    value_format,
    row_block,
    column_block,
    limits=None,
):
    """Draw one annotated evaluation matrix."""
    cmap = (
        plt.colormaps[color_map].copy()
        if isinstance(color_map, str)
        else color_map.copy()
    )
    cmap.set_bad("#d9d9d9")
    vmin, vmax = limits or (None, None)
    image = axis.imshow(
        np.ma.masked_invalid(values),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(column_labels)), column_labels)
    axis.set_yticks(np.arange(len(row_labels)), row_labels)
    axis.set_title(title)

    for separator in range(row_block, len(row_labels), row_block):
        axis.axhline(separator - 0.5, color="white", linewidth=2)
    for separator in range(column_block, len(column_labels), column_block):
        axis.axvline(separator - 0.5, color="white", linewidth=2)

    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        text = value_format.format(value) if np.isfinite(value) else "n/a"
        axis.text(
            column,
            row,
            text,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="semibold",
            color="black",
        )

    colorbar = axis.figure.colorbar(
        image,
        ax=axis,
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label(colorbar_label)
    return colorbar


def plot_point_robot_evaluation(
    dataframe,
    benchmark_groups,
    order_methods=_ORDER_METHODS,
):
    """Show success, path length, and planning time in compact matrices."""
    order_methods = _normalize_order_methods(order_methods)
    groups = list(benchmark_groups.items())
    if not groups:
        raise ValueError("At least one PointRobot environment is required.")

    selected = _select_results(
        dataframe,
        [
            benchmark_name
            for _, benchmark_names in groups
            for benchmark_name in benchmark_names
        ],
        order_methods,
    )
    goal_counts = sorted(selected["number_of_goals"].unique())
    method_order = [
        method
        for method in ("greedy", "exact")
        if method in order_methods
    ]
    planner_labels = dict(zip(_PLANNERS, _PLANNER_LABELS))
    columns = [
        (planner, method)
        for planner in _PLANNERS
        for method in method_order
    ]
    column_labels = [
        f"{planner_labels[planner]}\n{method.capitalize()}"
        for planner, method in columns
    ]

    rows = []
    for goal_count in goal_counts:
        for environment_label, benchmark_names in groups:
            matching_name = next(
                name
                for name in benchmark_names
                if int(
                    selected.loc[
                        selected["benchmark"] == name,
                        "number_of_goals",
                    ].iloc[0]
                )
                == goal_count
            )
            short_label = environment_label.split(" (")[0]
            rows.append(
                (matching_name, f"{short_label} – {goal_count}")
            )

    success_matrix = np.full((len(rows), len(columns)), np.nan)
    length_matrix = np.full((len(rows), len(columns)), np.nan)
    time_matrix = np.full((len(rows), len(columns)), np.nan)
    for row_index, (benchmark_name, _) in enumerate(rows):
        for column_index, (planner, method) in enumerate(columns):
            result = selected[
                (selected["benchmark"] == benchmark_name)
                & (selected["base_planner"] == planner)
                & (selected["order_method"] == method)
            ]
            if result.empty:
                continue
            success_matrix[row_index, column_index] = (
                result["success"].mean() * 100.0
            )
            length_matrix[row_index, column_index] = result.loc[
                result["success"],
                "final_path_length",
            ].mean()
            time_matrix[row_index, column_index] = (
                result["planning_time"].mean()
            )

    figure, axes = plt.subplots(3, 1, figsize=(12, 16))
    row_labels = [label for _, label in rows]
    common = {
        "row_labels": row_labels,
        "column_labels": column_labels,
        "row_block": len(groups),
        "column_block": len(method_order),
    }
    success_colorbar = _heatmap(
        axes[0],
        success_matrix,
        title="Roundtrip success rate",
        color_map=_POINT_HEATMAP_COLORS,
        colorbar_label="Success rate",
        value_format="{:.0f}%",
        limits=(0, 100),
        **common,
    )
    success_colorbar.set_ticks(
        [0, 20, 40, 60, 80, 100],
        labels=["0%", "20%", "40%", "60%", "80%", "100%"],
    )
    _heatmap(
        axes[1],
        length_matrix,
        title="Mean final roundtrip length (successful runs)",
        color_map=_POINT_HEATMAP_COLORS.reversed(),
        colorbar_label="Path length",
        value_format="{:.1f}",
        **common,
    )
    _heatmap(
        axes[2],
        time_matrix,
        title="Mean total planning time",
        color_map=_POINT_HEATMAP_COLORS.reversed(),
        colorbar_label="Planning time [s]",
        value_format="{:.2f} s",
        **common,
    )

    seed_count = selected["seed"].nunique()
    figure.suptitle(
        "PointRobot evaluation: "
        f"54 combinations, {seed_count} seeds each"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def plot_planar_robot_evaluation(
    dataframe,
    benchmark_groups,
    planner_name,
    order_method,
):
    """Compare the selected PlanarManipulator benchmark cases."""
    benchmark_names = [
        name
        for group_names in benchmark_groups.values()
        for name in group_names
    ]
    selected = _select_results(
        dataframe,
        benchmark_names,
        (order_method,),
    )
    selected = selected[
        selected["base_planner"] == planner_name
    ]
    if selected.empty:
        raise ValueError(
            f"No PlanarManipulator results exist for {planner_name!r}."
        )

    group_for_benchmark = {
        name: group
        for group, group_names in benchmark_groups.items()
        for name in group_names
    }
    selected["benchmark_group"] = selected["benchmark"].map(
        group_for_benchmark
    )
    values = selected.groupby("benchmark_group").agg(
        dof=("dof", "first"),
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length=("successful_path_length", "mean"),
        path_points=("successful_path_points", "mean"),
        roadmap_size=("roadmap_size", "mean"),
        collision_checks=("collision_checks", "mean"),
    ).reindex(benchmark_groups)
    values["success_rate"] *= 100.0
    values["normalized_path_length"] = (
        values["path_length"] / np.sqrt(values["dof"])
    )

    labels = [
        name.removeprefix("PlanarRobot ")
        for name in benchmark_groups
    ]
    positions = np.arange(len(benchmark_groups))
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))

    success_axis = axes[0, 0]
    success_bars = success_axis.bar(
        positions,
        values["success_rate"],
        color=_BLUE,
        edgecolor="black",
    )
    success_axis.set_xticks(positions, labels)
    _style_axis(success_axis, "Success rate [%]", _BLUE)
    success_axis.set_ylim(0, 110)
    _add_value_labels(success_axis, success_bars, "{:.0f}%")
    success_axis.set_title("Reliability")

    time_axis = axes[0, 1]
    time_axis.bar(
        positions,
        values["planning_time"],
        color=_BLUE,
        edgecolor="black",
    )
    time_axis.set_xticks(positions, labels)
    _style_axis(time_axis, "Planning time [s]", _BLUE)
    time_axis.set_title("Planning time")

    length_axis = axes[1, 0]
    points_axis = length_axis.twinx()
    width = 0.32
    length_axis.bar(
        positions - width / 2,
        values["normalized_path_length"],
        width,
        color=_YELLOW,
        edgecolor="black",
    )
    points_axis.bar(
        positions + width / 2,
        values["path_points"],
        width,
        color=_PURPLE,
        edgecolor="black",
    )
    length_axis.set_xticks(positions, labels)
    _style_axis(
        length_axis,
        r"Normalized path length [$\mathrm{rad}/\sqrt{DoF}$]",
        _YELLOW,
    )
    _style_right_axis(
        points_axis,
        "Points in final path",
        _PURPLE,
    )
    length_axis.legend(
        handles=_metric_handles(
            [
                ("Normalized path length", _YELLOW),
                ("Points in final path", _PURPLE),
            ]
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        title="Path quality",
    )

    roadmap_axis = axes[1, 1]
    checks_axis = roadmap_axis.twinx()
    roadmap_axis.bar(
        positions - width / 2,
        values["roadmap_size"],
        width,
        color=_PURPLE,
        edgecolor="black",
    )
    checks_axis.bar(
        positions + width / 2,
        values["collision_checks"],
        width,
        color=_YELLOW,
        edgecolor="black",
    )
    roadmap_axis.set_xticks(positions, labels)
    _style_axis(
        roadmap_axis,
        "Roadmap size [nodes + edges]",
        _PURPLE,
    )
    _style_right_axis(checks_axis, "Collision checks", _YELLOW)
    roadmap_axis.legend(
        handles=_metric_handles(
            [
                ("Roadmap size", _PURPLE),
                ("Collision checks", _YELLOW),
            ]
        ),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=1.0,
        edgecolor="black",
        title="Computational effort",
    )

    goal_count = int(selected["number_of_goals"].iloc[0])
    figure.suptitle(
        "PlanarManipulator evaluation - selected configuration cases\n"
        f"{planner_name}, {order_method}, {goal_count} goals"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0)
    return figure


def _planner_means(dataframe, benchmark_names, order_method):
    selected = _select_results(
        dataframe,
        benchmark_names,
        (order_method,),
    )
    selected = selected[
        selected["base_planner"].isin(_COMPARISON_PLANNERS)
    ]
    values = selected.groupby("base_planner").agg(
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length=("successful_path_length", "mean"),
        path_points=("successful_path_points", "mean"),
        successful_subpaths=("successful_subpaths", "mean"),
        failed_subpaths=("failed_subpaths", "mean"),
        roadmap_size=("roadmap_size", "mean"),
        collision_checks=("collision_checks", "mean"),
    ).reindex(_COMPARISON_PLANNERS)
    values["success_rate"] *= 100.0
    return values


def _planner_bars(
    axis,
    positions,
    values,
    metric,
    *,
    color,
    width,
    offset=0,
    bottom=None,
):
    return axis.bar(
        positions + offset,
        values[metric],
        width,
        bottom=values[bottom] if bottom else None,
        color=color,
        edgecolor="black",
        linewidth=0.6,
    )


def _plot_success_rate(values, title):
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(_COMPARISON_PLANNERS))
    bars = _planner_bars(
        axis,
        positions,
        values,
        "success_rate",
        color=_BLUE,
        width=0.55,
    )

    _set_planner_axis(axis, positions)
    _style_axis(axis, "Success rate [%]", _BLUE)
    axis.set_ylim(0, 110)
    _add_value_labels(axis, bars, "{:.0f}%")
    figure.suptitle(f"{title}: Reliability", y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    return figure


def _plot_path_and_time(values, title):
    figure, time_axis = plt.subplots(figsize=(12, 6))
    length_axis = time_axis.twinx()
    points_axis = time_axis.twinx()
    positions = np.arange(len(_COMPARISON_PLANNERS))
    width = 0.18

    _planner_bars(
        time_axis,
        positions,
        values,
        "planning_time",
        color=_BLUE,
        width=width,
        offset=-0.25,
    )
    _planner_bars(
        length_axis,
        positions,
        values,
        "path_length",
        color=_YELLOW,
        width=width,
    )
    _planner_bars(
        points_axis,
        positions,
        values,
        "path_points",
        color=_PURPLE,
        width=width,
        offset=0.25,
    )

    _set_planner_axis(time_axis, positions)
    _style_axis(time_axis, "Planning time [s]", _BLUE)
    _style_right_axis(length_axis, "Final roundtrip length", _YELLOW)
    _style_right_axis(
        points_axis,
        "Points in final path",
        _PURPLE,
        offset=70,
    )
    figure.legend(
        handles=_metric_handles(
            [
                ("Planning time", _BLUE),
                ("Final path length", _YELLOW),
                ("Points in final path", _PURPLE),
            ]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=3,
    )
    figure.suptitle(
        f"{title}: Planning effort and path quality",
        y=0.98,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.78,
        bottom=0.13,
        top=0.83,
    )
    return figure


def _plot_subpaths(values, title):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(_COMPARISON_PLANNERS))
    successful = _planner_bars(
        axis,
        positions,
        values,
        "successful_subpaths",
        color=_GREEN,
        width=0.55,
    )
    failed = _planner_bars(
        axis,
        positions,
        values,
        "failed_subpaths",
        color=_RED,
        width=0.55,
        bottom="successful_subpaths",
    )

    _set_planner_axis(axis, positions)
    _style_axis(axis, "Average number of pairwise paths", _BLUE)
    _add_segment_labels(axis, successful)
    _add_segment_labels(axis, failed)
    figure.legend(
        handles=_metric_handles(
            [
                ("Successful subpaths", _GREEN),
                ("Failed subpaths", _RED),
            ]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
    )
    figure.suptitle(f"{title}: Pairwise paths", y=0.98)
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.14,
        top=0.82,
    )
    return figure


def _plot_roadmap_and_checks(values, title):
    figure, roadmap_axis = plt.subplots(figsize=(11, 5.5))
    checks_axis = roadmap_axis.twinx()
    positions = np.arange(len(_COMPARISON_PLANNERS))
    width = 0.3

    _planner_bars(
        roadmap_axis,
        positions,
        values,
        "roadmap_size",
        color=_PURPLE,
        width=width,
        offset=-0.18,
    )
    _planner_bars(
        checks_axis,
        positions,
        values,
        "collision_checks",
        color=_YELLOW,
        width=width,
        offset=0.18,
    )

    _set_planner_axis(roadmap_axis, positions)
    _style_axis(
        roadmap_axis,
        "Roadmap size [nodes + edges]",
        _PURPLE,
    )
    _style_right_axis(checks_axis, "Collision checks", _YELLOW)
    figure.legend(
        handles=_metric_handles(
            [
                ("Roadmap size", _PURPLE),
                ("Collision checks", _YELLOW),
            ]
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
    )
    figure.suptitle(
        f"{title}: Roadmap and collision-checking effort",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.84))
    return figure


def plot_base_planner_comparison(
    dataframe,
    benchmark_names,
    order_method="exact",
    title="PointRobot base-planner comparison",
):
    """Compare the planners with one fixed visit-order method."""
    values = _planner_means(
        dataframe,
        benchmark_names,
        order_method,
    )
    return {
        "success_rate": _plot_success_rate(values, title),
        "path_and_time": _plot_path_and_time(values, title),
        "subpaths": _plot_subpaths(values, title),
        "roadmap_and_checks": _plot_roadmap_and_checks(values, title),
    }
