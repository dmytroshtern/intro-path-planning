"""Static visualizations for roundtrip-planning results."""

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from notebooks.IPEnvironmentKin import KinChainCollisionChecker


_PAIRWISE_COLOR = "#9aa0a6"
_TOUR_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
)


def _require_success(result):
    """Raise a clear error when a failed roundtrip is visualized."""
    if not result.get("success", False):
        reason = result.get("reason", "unknown reason")
        raise ValueError(f"Cannot visualize a failed roundtrip: {reason}")


def _configure_axis(benchmark, ax):
    environment = benchmark.collisionChecker
    limits = environment.getEnvironmentLimits()
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)

    if isinstance(environment, KinChainCollisionChecker):
        ax.set_xlabel(r"$q_1$ [rad]")
        ax.set_ylabel(r"$q_2$ [rad]")
    else:
        environment.drawObstacles(ax)
        ax.set_xlabel("x")
        ax.set_ylabel("y")


def _draw_required_configurations(benchmark, ax, show_labels=True):
    start = np.asarray(benchmark.startList[0], dtype=float)
    goals = np.asarray(benchmark.goalList, dtype=float)

    ax.scatter(
        start[0],
        start[1],
        marker="*",
        s=220,
        color="#2ca02c",
        edgecolor="black",
        zorder=20,
        label="Start",
    )
    ax.scatter(
        goals[:, 0],
        goals[:, 1],
        marker="X",
        s=100,
        color="#d62728",
        edgecolor="black",
        zorder=20,
        label="Goals",
    )

    if show_labels:
        ax.annotate("S", start, xytext=(7, 7), textcoords="offset points")
        for index, goal in enumerate(goals, start=1):
            ax.annotate(
                f"G{index}",
                goal,
                xytext=(7, 7),
                textcoords="offset points",
            )


def _draw_path(
    ax,
    path,
    *,
    color,
    linewidth=2.0,
    alpha=1.0,
    label=None,
    arrows=False,
):
    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        return

    ax.plot(
        points[:, 0],
        points[:, 1],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        label=label,
        zorder=8,
    )
    if arrows:
        for first, second in zip(points[:-1], points[1:]):
            midpoint = 0.5 * (first + second)
            delta = 0.12 * (second - first)
            ax.annotate(
                "",
                xy=midpoint + delta,
                xytext=midpoint - delta,
                arrowprops={
                    "arrowstyle": "->",
                    "color": color,
                    "lw": max(1.0, linewidth - 0.5),
                },
                zorder=9,
            )


def plot_roundtrip_components(result, benchmark):
    """Show the required 2-DoF roundtrip elements in one figure."""
    if not result.get("success", False):
        reason = result.get("reason", "unknown reason")
        raise ValueError(f"Cannot visualize a failed roundtrip: {reason}")
    if benchmark.collisionChecker.getDim() != 2:
        raise ValueError("Roundtrip component plots require 2 DoF.")

    figure, axes = plt.subplots(1, 2, figsize=(16, 7))
    pairwise_ax, tour_ax = axes

    _configure_axis(benchmark, pairwise_ax)
    pairwise_results = [
        pair_result
        for pair_result in result.get("pairwise_results", {}).values()
        if pair_result.get("success", False)
        and not pair_result.get("metadata", {}).get("reversed", False)
    ]
    for index, pair_result in enumerate(pairwise_results):
        _draw_path(
            pairwise_ax,
            pair_result["path_configs"],
            color=_PAIRWISE_COLOR,
            linewidth=1.2,
            alpha=0.55,
            label="Successful pairwise paths" if index == 0 else None,
        )
    for index, pair in enumerate(result["used_pairs"]):
        pair_result = result["pairwise_results"][tuple(pair)]
        _draw_path(
            pairwise_ax,
            pair_result["path_configs"],
            color="#4c78a8",
            linewidth=2.4,
            alpha=0.9,
            label="Selected subpaths" if index == 0 else None,
        )
    _draw_required_configurations(benchmark, pairwise_ax)
    pairwise_ax.set_title("Planned pairwise subpaths")
    pairwise_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8,
        frameon=True,
    )

    _configure_axis(benchmark, tour_ax)
    _draw_path(
        tour_ax,
        result["final_path_configs"],
        color="#8bbce5",
        linewidth=6.0,
        alpha=0.55,
        label="Final combined path",
    )
    for index, pair in enumerate(result["used_pairs"]):
        pair_result = result["pairwise_results"][tuple(pair)]
        color = _TOUR_COLORS[index % len(_TOUR_COLORS)]
        _draw_path(
            tour_ax,
            pair_result["path_configs"],
            color=color,
            linewidth=2.6,
            label=f"{pair[0]} → {pair[1]}",
            arrows=True,
        )
    _draw_required_configurations(benchmark, tour_ax)
    tour_ax.set_title(
        "Selected roundtrip: "
        + " → ".join(result["visit_order"])
        + f"\nFinal path length: {result['tour_cost']:.2f}"
    )
    tour_ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        fontsize=8,
        ncol=3,
        frameon=True,
    )

    figure.suptitle(benchmark.name, fontsize=15)
    figure.tight_layout(rect=(0, 0.12, 1, 0.96))
    return figure, axes


def plot_metagraph(planner, result, ax=None):
    """Draw the metagraph and highlight the selected tour."""
    _require_success(result)
    graph = getattr(planner, "metagraph", None)
    if graph is None:
        raise ValueError("The planner does not expose a metagraph.")

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    positions = nx.circular_layout(graph)
    selected_edges = {
        tuple(pair) for pair in result["used_pairs"]
    }
    if not graph.is_directed():
        selected_edges |= {(target, source) for source, target in selected_edges}

    regular_edges = [
        edge for edge in graph.edges if tuple(edge) not in selected_edges
    ]
    tour_edges = [
        edge for edge in graph.edges if tuple(edge) in selected_edges
    ]

    node_colors = [
        "#2ca02c" if node == "S" else "#d62728"
        for node in graph.nodes
    ]
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=900,
        edgecolors="black",
        ax=ax,
    )
    nx.draw_networkx_labels(graph, positions, font_color="white", ax=ax)
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=regular_edges,
        edge_color="#b7b7b7",
        width=1.2,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=tour_edges,
        edge_color="#0057b8",
        width=3.0,
        arrows=graph.is_directed(),
        ax=ax,
    )
    edge_labels = {
        edge: f"{data.get('weight', data.get('cost', 0.0)):.1f}"
        for *edge_nodes, data in graph.edges(data=True)
        for edge in [tuple(edge_nodes)]
    }
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.35,
        rotate=False,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.85,
        },
        ax=ax,
    )
    ax.set_title("Metagraph and selected roundtrip")
    ax.axis("off")
    return ax
