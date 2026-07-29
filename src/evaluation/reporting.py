"""Plots for the PointRobot and PlanarManipulator evaluations."""

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_PLANNERS = ("BasicPRM", "VisibilityPRM", "LazyPRM")
_PLANNER_LABELS = _PLANNERS
_ORDER_METHODS = ("exact", "greedy")

_BLUE = "#0000ff"
_YELLOW = "#d6cf00"
_PURPLE = "#800080"
_GREEN = "#2ca02c"
_RED = "#d62728"
_METHOD_HATCHES = {"exact": "", "greedy": "//"}


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


def _method_handles(order_methods):
    if len(order_methods) == 1:
        return []
    return [
        Patch(
            facecolor="white",
            edgecolor="black",
            hatch=_METHOD_HATCHES[method],
            label=method.capitalize(),
        )
        for method in order_methods
    ]


def _metric_handles(metrics):
    return [
        Patch(facecolor=color, edgecolor="black", label=label)
        for label, color in metrics
    ]


def _set_planner_axis(axis, positions):
    axis.set_xticks(positions, _PLANNER_LABELS)
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


def plot_point_robot_evaluation(
    dataframe,
    benchmark_groups,
    order_methods=_ORDER_METHODS,
):
    """Show all PointRobot combinations in two compact matrices."""
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

    figure, axes = plt.subplots(2, 1, figsize=(12, 11))
    success_image = axes[0].imshow(
        success_matrix,
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        aspect="auto",
    )
    length_colormap = plt.colormaps["viridis"].copy()
    length_colormap.set_bad("#d9d9d9")
    length_image = axes[1].imshow(
        np.ma.masked_invalid(length_matrix),
        cmap=length_colormap,
        aspect="auto",
    )

    row_labels = [label for _, label in rows]
    for axis, title in zip(
        axes,
        (
            "Roundtrip success rate",
            "Mean final roundtrip length (successful runs)",
        ),
    ):
        axis.set_xticks(
            np.arange(len(column_labels)),
            column_labels,
        )
        axis.set_yticks(
            np.arange(len(row_labels)),
            row_labels,
        )
        axis.set_title(title)
        for separator in range(len(groups), len(rows), len(groups)):
            axis.axhline(
                separator - 0.5,
                color="white",
                linewidth=2,
            )
        for separator in range(2, len(column_labels), 2):
            axis.axvline(
                separator - 0.5,
                color="white",
                linewidth=2,
            )

    for row_index in range(success_matrix.shape[0]):
        for column_index in range(success_matrix.shape[1]):
            success_rate = success_matrix[row_index, column_index]
            axes[0].text(
                column_index,
                row_index,
                f"{success_rate:.0f}%"
                if np.isfinite(success_rate)
                else "n/a",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
            path_length = length_matrix[row_index, column_index]
            axes[1].text(
                column_index,
                row_index,
                f"{path_length:.1f}"
                if np.isfinite(path_length)
                else "n/a",
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "black"
                    if np.isfinite(path_length)
                    and length_image.norm(path_length) > 0.60
                    else "white"
                    if np.isfinite(path_length)
                    else "black"
                ),
            )

    success_colorbar = figure.colorbar(
        success_image,
        ax=axes[0],
        fraction=0.025,
        pad=0.02,
    )
    success_colorbar.set_ticks(
        [0, 20, 40, 60, 80, 100],
        labels=["0%", "20%", "40%", "60%", "80%", "100%"],
    )
    success_colorbar.set_label("Success rate")
    length_colorbar = figure.colorbar(
        length_image,
        ax=axes[1],
        fraction=0.025,
        pad=0.02,
    )
    length_colorbar.set_label("Path length")
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
    """Compare PlanarManipulator cases across three configuration sets."""
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
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length=("successful_path_length", "mean"),
        path_points=("successful_path_points", "mean"),
        roadmap_size=("roadmap_size", "mean"),
        collision_checks=("collision_checks", "mean"),
        difficulty=("difficulty", "first"),
    ).reindex(benchmark_groups)
    values["success_rate"] *= 100.0

    labels = [
        f"{name.removeprefix('PlanarRobot ')}\n"
        f"(difficulty {int(values.loc[name, 'difficulty'])})"
        for name in benchmark_groups
    ]
    positions = np.arange(len(benchmark_groups))
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

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
        values["path_length"],
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
    _style_axis(length_axis, "Final roundtrip length", _YELLOW)
    _style_right_axis(
        points_axis,
        "Points in final path",
        _PURPLE,
    )
    length_axis.legend(
        handles=_metric_handles(
            [
                ("Final path length", _YELLOW),
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
        f"PlanarManipulator evaluation — {planner_name}, "
        f"{order_method}, {goal_count} goals, "
        "3 configuration sets each"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0)
    return figure


def _planner_means(dataframe, benchmark_names, order_methods):
    selected = _select_results(
        dataframe,
        benchmark_names,
        order_methods,
    )
    values = selected.groupby(
        ["base_planner", "order_method"]
    ).agg(
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length=("successful_path_length", "mean"),
        path_points=("successful_path_points", "mean"),
        successful_subpaths=("successful_subpaths", "mean"),
        failed_subpaths=("failed_subpaths", "mean"),
        roadmap_size=("roadmap_size", "mean"),
        collision_checks=("collision_checks", "mean"),
    )
    index = pd.MultiIndex.from_product(
        [_PLANNERS, order_methods],
        names=["base_planner", "order_method"],
    )
    values = values.reindex(index)
    values["success_rate"] *= 100.0
    return values


def _values_for(values, metric, order_method):
    return (
        values.xs(order_method, level="order_method")[metric]
        .reindex(_PLANNERS)
        .to_numpy(dtype=float)
    )


def _bars_for_methods(
    axis,
    values,
    metric,
    positions,
    centers,
    width,
    color,
    order_methods,
    stacked_on=None,
):
    bars = []
    offsets = (0,) if len(order_methods) == 1 else (-width / 2, width / 2)
    for offset, method in zip(offsets, order_methods):
        bottom = (
            _values_for(values, stacked_on, method)
            if stacked_on
            else None
        )
        bars.append(
            axis.bar(
                positions + centers + offset,
                _values_for(values, metric, method),
                width,
                bottom=bottom,
                color=color,
                edgecolor="black",
                linewidth=0.6,
                hatch=_METHOD_HATCHES[method],
            )
        )
    return bars


def _plot_success_rate(values, title, order_methods):
    figure, axis = plt.subplots(figsize=(9, 5))
    positions = np.arange(len(_PLANNERS))
    bars = _bars_for_methods(
        axis,
        values,
        "success_rate",
        positions,
        centers=0,
        width=0.32,
        color=_BLUE,
        order_methods=order_methods,
    )

    _set_planner_axis(axis, positions)
    _style_axis(axis, "Success rate [%]", _BLUE)
    axis.set_ylim(0, 110)
    for method_bars in bars:
        _add_value_labels(axis, method_bars, "{:.0f}%")
    if len(order_methods) > 1:
        figure.legend(
            handles=_method_handles(order_methods),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=len(order_methods),
            title="Visit order",
        )
    figure.suptitle(f"{title}: Reliability", y=0.98)
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.14,
        top=0.78,
    )
    return figure


def _plot_path_and_time(values, title, order_methods):
    figure, time_axis = plt.subplots(figsize=(12, 6))
    length_axis = time_axis.twinx()
    points_axis = time_axis.twinx()
    positions = np.arange(len(_PLANNERS))
    width = 0.12

    _bars_for_methods(
        time_axis,
        values,
        "planning_time",
        positions,
        centers=-0.28,
        width=width,
        color=_BLUE,
        order_methods=order_methods,
    )
    _bars_for_methods(
        length_axis,
        values,
        "path_length",
        positions,
        centers=0,
        width=width,
        color=_YELLOW,
        order_methods=order_methods,
    )
    _bars_for_methods(
        points_axis,
        values,
        "path_points",
        positions,
        centers=0.28,
        width=width,
        color=_PURPLE,
        order_methods=order_methods,
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
        handles=(
            _metric_handles(
                [
                    ("Planning time", _BLUE),
                    ("Final path length", _YELLOW),
                    ("Points in final path", _PURPLE),
                ]
            )
            + _method_handles(order_methods)
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=5,
    )
    figure.suptitle(
        f"{title}: Planning effort and path quality",
        y=0.98,
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.78,
        bottom=0.13,
        top=0.78,
    )
    return figure


def _plot_subpaths(values, title, order_methods):
    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(_PLANNERS))
    width = 0.32
    successful = _bars_for_methods(
        axis,
        values,
        "successful_subpaths",
        positions,
        centers=0,
        width=width,
        color=_GREEN,
        order_methods=order_methods,
    )
    failed = _bars_for_methods(
        axis,
        values,
        "failed_subpaths",
        positions,
        centers=0,
        width=width,
        color=_RED,
        order_methods=order_methods,
        stacked_on="successful_subpaths",
    )

    _set_planner_axis(axis, positions)
    _style_axis(axis, "Average number of pairwise paths", _BLUE)
    for method_bars in successful + failed:
        _add_segment_labels(axis, method_bars)
    figure.legend(
        handles=(
            _metric_handles(
                [
                    ("Successful subpaths", _GREEN),
                    ("Failed subpaths", _RED),
                ]
            )
            + _method_handles(order_methods)
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
    )
    figure.suptitle(f"{title}: Pairwise paths", y=0.98)
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.14,
        top=0.76,
    )
    return figure


def _plot_roadmap_and_checks(values, title, order_methods):
    figure, roadmap_axis = plt.subplots(figsize=(11, 5.5))
    checks_axis = roadmap_axis.twinx()
    positions = np.arange(len(_PLANNERS))
    width = 0.17

    _bars_for_methods(
        roadmap_axis,
        values,
        "roadmap_size",
        positions,
        centers=-0.22,
        width=width,
        color=_PURPLE,
        order_methods=order_methods,
    )
    _bars_for_methods(
        checks_axis,
        values,
        "collision_checks",
        positions,
        centers=0.22,
        width=width,
        color=_YELLOW,
        order_methods=order_methods,
    )

    _set_planner_axis(roadmap_axis, positions)
    _style_axis(
        roadmap_axis,
        "Roadmap size [nodes + edges]",
        _PURPLE,
    )
    _style_right_axis(checks_axis, "Collision checks", _YELLOW)
    figure.legend(
        handles=(
            _metric_handles(
                [
                    ("Roadmap size", _PURPLE),
                    ("Collision checks", _YELLOW),
                ]
            )
            + _method_handles(order_methods)
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=4,
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
    order_methods=_ORDER_METHODS,
    title="PointRobot base-planner comparison",
):
    """Compare all three base planners using the PointRobot results."""
    order_methods = _normalize_order_methods(order_methods)
    values = _planner_means(
        dataframe,
        benchmark_names,
        order_methods,
    )
    return {
        "success_rate": _plot_success_rate(
            values,
            title,
            order_methods,
        ),
        "path_and_time": _plot_path_and_time(
            values,
            title,
            order_methods,
        ),
        "subpaths": _plot_subpaths(
            values,
            title,
            order_methods,
        ),
        "roadmap_and_checks": _plot_roadmap_and_checks(
            values,
            title,
            order_methods,
        ),
    }
