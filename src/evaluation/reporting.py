"""Plots used in the roundtrip evaluation notebook."""

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np


_PLANNERS = ("BasicPRM", "VisibilityPRM", "LazyPRM")
_ALL_PLANNERS = _PLANNERS + ("MultiQueryPRM",)
_BLUE = "#0000ff"
_YELLOW = "#d6cf00"
_PURPLE = "#800080"
_GREEN = "#2ca02c"
_RED = "#d62728"
_HEATMAP = LinearSegmentedColormap.from_list(
    "evaluation",
    plt.colormaps["RdYlGn"](np.linspace(0.15, 0.85, 256)),
)


def _select(dataframe, benchmark_names, order_methods=None):
    """Select experiment rows and add metrics used by several plots."""
    names = (
        (benchmark_names,)
        if isinstance(benchmark_names, str)
        else tuple(benchmark_names)
    )
    selected = dataframe[dataframe["benchmark"].isin(names)].copy()
    if order_methods is not None:
        methods = (
            (order_methods,)
            if isinstance(order_methods, str)
            else tuple(order_methods)
        )
        selected = selected[selected["order_method"].isin(methods)]
    if selected.empty:
        raise ValueError("No matching benchmark results are available.")

    selected["successful_length"] = selected[
        "final_path_length"
    ].where(selected["success"])
    selected["successful_points"] = selected[
        "final_path_points"
    ].where(selected["success"])
    selected["roadmap_size"] = (
        selected["roadmap_nodes"] + selected["roadmap_edges"]
    )
    return selected


def _style_axis(axis, label, color, right=False, offset=0):
    side = "right" if right else "left"
    if right and offset:
        axis.spines["right"].set_position(("outward", offset))
    axis.set_ylabel(label, color=color)
    axis.tick_params(axis="y", labelcolor=color)
    axis.spines[side].set_color(color)
    if not right:
        axis.grid(axis="y", alpha=0.25)


def _legend(metrics):
    return [
        Patch(facecolor=color, edgecolor="black", label=label)
        for _, label, _, color in metrics
    ]


def _label_bars(axis, bars, template):
    labels = [
        template.format(bar.get_height())
        if np.isfinite(bar.get_height())
        else "n/a"
        for bar in bars
    ]
    axis.bar_label(bars, labels=labels, padding=3, fontsize=8)


def _metric_bars(
    axis,
    positions,
    labels,
    values,
    metrics,
    *,
    legend_title=None,
):
    """Plot one to three metrics with separate y-axes."""
    count = len(metrics)
    width = min(0.64 / count, 0.32)
    offsets = (np.arange(count) - (count - 1) / 2) * width
    axes = [axis]
    bars = []

    for index, (column, _, ylabel, color) in enumerate(metrics):
        current = axis if index == 0 else axis.twinx()
        if index:
            axes.append(current)
        bars.append(
            current.bar(
                positions + offsets[index],
                values[column],
                width,
                color=color,
                edgecolor="black",
            )
        )
        _style_axis(
            current,
            ylabel,
            color,
            right=index > 0,
            offset=70 * max(0, index - 1),
        )

    axis.set_xticks(positions, labels)
    if count > 1:
        axis.legend(
            handles=_legend(metrics),
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=count,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            title=legend_title,
        )
    return axes, bars


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
    row_separators=(),
    column_block=1,
    limits=None,
):
    """Draw one annotated result matrix."""
    cmap = color_map.copy()
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

    for separator in row_separators:
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
    order_methods=("exact", "greedy"),
):
    """Show success, path length, and planning time as matrices."""
    requested = (
        (order_methods,)
        if isinstance(order_methods, str)
        else tuple(order_methods)
    )
    methods = tuple(
        method for method in ("greedy", "exact") if method in requested
    )
    names = [
        name
        for group_names in benchmark_groups.values()
        for name in group_names
    ]
    selected = _select(dataframe, names, methods)
    selected = selected[selected["base_planner"].isin(_PLANNERS)]

    rows = []
    for goals in sorted(selected["number_of_goals"].unique()):
        for group, group_names in benchmark_groups.items():
            for name in group_names:
                match = selected[selected["benchmark"] == name]
                if (
                    not match.empty
                    and int(match["number_of_goals"].iloc[0]) == goals
                ):
                    rows.append(
                        (name, f"{group.split(' (')[0]} – {goals}", goals)
                    )
                    break

    columns = [
        (planner, method)
        for planner in _PLANNERS
        for method in methods
    ]
    summary = selected.groupby(
        ["benchmark", "base_planner", "order_method"]
    ).agg(
        success_rate=("success", "mean"),
        path_length=("successful_length", "mean"),
        planning_time=("planning_time", "mean"),
    )
    summary["success_rate"] *= 100.0

    def matrix(metric):
        result = np.full((len(rows), len(columns)), np.nan)
        for row, (benchmark, _, _) in enumerate(rows):
            for column, (planner, method) in enumerate(columns):
                key = (benchmark, planner, method)
                if key in summary.index:
                    result[row, column] = summary.loc[key, metric]
        return result

    row_labels = [label for _, label, _ in rows]
    column_labels = [
        f"{planner}\n{method.capitalize()}"
        for planner, method in columns
    ]
    row_separators = [
        index
        for index in range(1, len(rows))
        if rows[index][2] != rows[index - 1][2]
    ]
    common = dict(
        row_labels=row_labels,
        column_labels=column_labels,
        row_separators=row_separators,
        column_block=len(methods),
    )

    figure, axes = plt.subplots(3, 1, figsize=(12, 16))
    success_bar = _heatmap(
        axes[0],
        matrix("success_rate"),
        title="Roundtrip success rate",
        color_map=_HEATMAP,
        colorbar_label="Success rate",
        value_format="{:.0f}%",
        limits=(0, 100),
        **common,
    )
    success_bar.set_ticks(
        [0, 20, 40, 60, 80, 100],
        labels=["0%", "20%", "40%", "60%", "80%", "100%"],
    )
    _heatmap(
        axes[1],
        matrix("path_length"),
        title="Mean final roundtrip length (successful runs)",
        color_map=_HEATMAP.reversed(),
        colorbar_label="Path length",
        value_format="{:.1f}",
        **common,
    )
    _heatmap(
        axes[2],
        matrix("planning_time"),
        title="Mean total planning time",
        color_map=_HEATMAP.reversed(),
        colorbar_label="Planning time [s]",
        value_format="{:.2f} s",
        **common,
    )

    combinations = selected[
        ["benchmark", "base_planner", "order_method"]
    ].drop_duplicates().shape[0]
    seeds = selected["seed"].nunique()
    figure.suptitle(
        f"PointRobot evaluation: {combinations} combinations, "
        f"{seeds} seed{'s' if seeds != 1 else ''} each"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return figure


def plot_planar_robot_evaluation(
    dataframe,
    benchmark_groups,
    planner_name,
    order_method,
):
    """Compare the selected PlanarManipulator configuration groups."""
    names = [
        name
        for group_names in benchmark_groups.values()
        for name in group_names
    ]
    selected = _select(dataframe, names, order_method)
    selected = selected[selected["base_planner"] == planner_name]
    group_for_name = {
        name: group
        for group, group_names in benchmark_groups.items()
        for name in group_names
    }
    selected["group"] = selected["benchmark"].map(group_for_name)
    values = selected.groupby("group").agg(
        dof=("dof", "first"),
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length=("successful_length", "mean"),
        path_points=("successful_points", "mean"),
        roadmap_size=("roadmap_size", "mean"),
        collision_checks=("collision_checks", "mean"),
    ).reindex(benchmark_groups)
    values["success_rate"] *= 100.0
    values["normalized_length"] = (
        values["path_length"] / np.sqrt(values["dof"])
    )

    labels = [
        name.removeprefix("PlanarRobot ")
        for name in benchmark_groups
    ]
    positions = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(14, 8))

    _, bars = _metric_bars(
        axes[0, 0],
        positions,
        labels,
        values,
        [("success_rate", "Success rate", "Success rate [%]", _BLUE)],
    )
    axes[0, 0].set_ylim(0, 110)
    axes[0, 0].set_title("Reliability")
    _label_bars(axes[0, 0], bars[0], "{:.0f}%")

    _metric_bars(
        axes[0, 1],
        positions,
        labels,
        values,
        [("planning_time", "Planning time", "Planning time [s]", _BLUE)],
    )
    axes[0, 1].set_title("Planning time")

    _metric_bars(
        axes[1, 0],
        positions,
        labels,
        values,
        [
            (
                "normalized_length",
                "Normalized path length",
                r"Normalized path length [$\mathrm{rad}/\sqrt{DoF}$]",
                _YELLOW,
            ),
            (
                "path_points",
                "Points in final path",
                "Points in final path",
                _PURPLE,
            ),
        ],
        legend_title="Path quality",
    )
    _metric_bars(
        axes[1, 1],
        positions,
        labels,
        values,
        [
            (
                "roadmap_size",
                "Roadmap size",
                "Roadmap size [nodes + edges]",
                _PURPLE,
            ),
            (
                "collision_checks",
                "Collision checks",
                "Collision checks",
                _YELLOW,
            ),
        ],
        legend_title="Computational effort",
    )

    goals = int(selected["number_of_goals"].iloc[0])
    figure.suptitle(
        "PlanarManipulator evaluation - selected configuration cases\n"
        f"{planner_name}, {order_method}, {goals} goals"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94), h_pad=2.0)
    return figure


def _planner_values(dataframe, benchmark_names, order_method):
    selected = _select(dataframe, benchmark_names, order_method)
    selected = selected[
        selected["base_planner"].isin(_ALL_PLANNERS)
    ]
    values = selected.groupby("base_planner").agg(
        success_rate=("success", "mean"),
        planning_time=("planning_time", "mean"),
        path_length=("successful_length", "mean"),
        path_points=("successful_points", "mean"),
        successful_subpaths=("successful_subpaths", "mean"),
        failed_subpaths=("failed_subpaths", "mean"),
        roadmap_size=("roadmap_size", "mean"),
        collision_checks=("collision_checks", "mean"),
    ).reindex(_ALL_PLANNERS)
    values["success_rate"] *= 100.0
    return values


def _metric_figure(values, metrics, title, figsize=(11, 5.5)):
    figure, axis = plt.subplots(figsize=figsize)
    positions = np.arange(len(values))
    axes, bars = _metric_bars(
        axis,
        positions,
        _ALL_PLANNERS,
        values,
        metrics,
    )
    axis.set_xlabel("Base planner")
    figure.suptitle(title, y=0.98)
    right = 0.78 if len(metrics) == 3 else 0.90
    figure.subplots_adjust(
        left=0.09,
        right=right,
        bottom=0.14,
        top=0.82 if len(metrics) > 1 else 0.90,
    )
    return figure, axes, bars


def plot_base_planner_comparison(
    dataframe,
    benchmark_names,
    order_method="exact",
    title="PointRobot base-planner comparison",
):
    """Compare all planners using one fixed visit-order method."""
    values = _planner_values(dataframe, benchmark_names, order_method)

    reliability, axes, bars = _metric_figure(
        values,
        [("success_rate", "Success rate", "Success rate [%]", _BLUE)],
        f"{title}: Reliability",
        (9, 5),
    )
    axes[0].set_ylim(0, 110)
    _label_bars(axes[0], bars[0], "{:.0f}%")

    path_and_time, _, _ = _metric_figure(
        values,
        [
            (
                "planning_time",
                "Planning time",
                "Planning time [s]",
                _BLUE,
            ),
            (
                "path_length",
                "Final path length",
                "Final roundtrip length",
                _YELLOW,
            ),
            (
                "path_points",
                "Points in final path",
                "Points in final path",
                _PURPLE,
            ),
        ],
        f"{title}: Planning effort and path quality",
        (12, 6),
    )

    subpaths, axis = plt.subplots(figsize=(9, 5.5))
    positions = np.arange(len(values))
    successful = axis.bar(
        positions,
        values["successful_subpaths"],
        0.55,
        color=_GREEN,
        edgecolor="black",
    )
    failed = axis.bar(
        positions,
        values["failed_subpaths"],
        0.55,
        bottom=values["successful_subpaths"],
        color=_RED,
        edgecolor="black",
    )
    axis.set_xticks(positions, _ALL_PLANNERS)
    axis.set_xlabel("Base planner")
    _style_axis(axis, "Average number of pairwise paths", _BLUE)
    for bar_group in (successful, failed):
        axis.bar_label(
            bar_group,
            labels=[
                f"{bar.get_height():.1f}" if bar.get_height() > 0 else ""
                for bar in bar_group
            ],
            label_type="center",
            fontsize=8,
        )
    subpaths.legend(
        handles=[
            Patch(facecolor=_GREEN, edgecolor="black",
                  label="Successful subpaths"),
            Patch(facecolor=_RED, edgecolor="black",
                  label="Failed subpaths"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
    )
    subpaths.suptitle(f"{title}: Pairwise paths", y=0.98)
    subpaths.subplots_adjust(
        left=0.10,
        right=0.97,
        bottom=0.14,
        top=0.82,
    )

    roadmap, _, _ = _metric_figure(
        values,
        [
            (
                "roadmap_size",
                "Roadmap size",
                "Roadmap size [nodes + edges]",
                _PURPLE,
            ),
            (
                "collision_checks",
                "Collision checks",
                "Collision checks",
                _YELLOW,
            ),
        ],
        f"{title}: Roadmap and collision-checking effort",
    )

    return {
        "success_rate": reliability,
        "path_and_time": path_and_time,
        "subpaths": subpaths,
        "roadmap_and_checks": roadmap,
    }
