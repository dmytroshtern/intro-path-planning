"""Static visualizations for roundtrip-planning results."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from notebooks.IPEnvironmentKin import KinChainCollisionChecker


PAIRWISE_COLOR = "#9aa0a6"
SELECTED_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
)


def _require_success(result: dict[str, Any]) -> None:
    if not result.get("success", False):
        reason = result.get("reason", "unknown reason")
        raise ValueError(f"Cannot visualize a failed roundtrip: {reason}")


def _configure_axis(benchmark, ax) -> None:
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


def _draw_required_configurations(benchmark, ax, show_labels=True) -> None:
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
) -> None:
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


def _successful_pairwise_results(result):
    """Yield successful planner calls without generated reverse copies."""
    for pair_result in result.get("pairwise_results", {}).values():
        metadata = pair_result.get("metadata", {})
        if metadata.get("reversed", False):
            continue
        if pair_result.get("success", False):
            yield pair_result


def _draw_pairwise_paths(result, benchmark, ax):
    _configure_axis(benchmark, ax)
    for index, pair_result in enumerate(
        _successful_pairwise_results(result)
    ):
        _draw_path(
            ax,
            pair_result["path_configs"],
            color=PAIRWISE_COLOR,
            linewidth=1.2,
            alpha=0.55,
            label="Successful pairwise paths" if index == 0 else None,
        )
    _draw_required_configurations(benchmark, ax)
    ax.set_title("Planned pairwise subpaths")
    ax.legend(loc="best")


def _draw_selected_tour(result, benchmark, ax):
    _configure_axis(benchmark, ax)
    _draw_path(
        ax,
        result["final_path_configs"],
        color="#8bbce5",
        linewidth=6.0,
        alpha=0.55,
        label="Final combined path",
    )

    for index, pair in enumerate(result["used_pairs"]):
        pair_result = result["pairwise_results"][tuple(pair)]
        color = SELECTED_COLORS[index % len(SELECTED_COLORS)]
        _draw_path(
            ax,
            pair_result["path_configs"],
            color=color,
            linewidth=2.6,
            label=f"{pair[0]} → {pair[1]}",
            arrows=True,
        )
    _draw_required_configurations(benchmark, ax)
    ax.set_title(
        "Selected roundtrip: "
        + " → ".join(result["visit_order"])
        + f"\nFinal path length: {result['tour_cost']:.2f}"
    )
    ax.legend(loc="best", fontsize=8, ncol=2)


def plot_roundtrip_components(result, benchmark):
    """Show the required 2-DoF roundtrip elements in one figure."""
    _require_success(result)
    if benchmark.collisionChecker.getDim() != 2:
        raise ValueError("Roundtrip component plots require 2 DoF.")

    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    _draw_pairwise_paths(result, benchmark, axes[0])
    _draw_selected_tour(result, benchmark, axes[1])
    figure.suptitle(benchmark.name, fontsize=15)
    figure.tight_layout()
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
        ax=ax,
    )
    ax.set_title("Metagraph and selected roundtrip")
    ax.axis("off")
    return ax
