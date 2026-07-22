"""
Pairwise path planning for the Roundtrip Planner.

This module calls a base planner, such as BasicPRM, KClosestPRM, LazyPRM,
or VisibilityPRM, between every pair of required configurations.

The base planner returns internal node IDs.

This module converts those node IDs into actual configurations and computes
a consistent path cost.
"""

from __future__ import annotations

import time
from itertools import combinations, permutations
from typing import Any

from .costs import path_length
from .result import pairwise_failure, pairwise_success


def compute_pairwise_paths(
    points: dict[str, Any],
    collision_checker: Any,
    config: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    """
    Compute pairwise paths between all required points.

    This function is designed to be passed as:

        config["pairwise_provider"] = compute_pairwise_paths

    Required config entries
    -----------------------
    config["base_planner_class"]:
        Planner class, for example BasicPRM, KClosestPRM, LazyPRM, ...

    config["base_planner_config"]:
        Config dictionary for the base planner.

    Optional config entries
    -----------------------
    config["directed"]:
        If True, compute all ordered pairs.
        If False, compute each unordered pair once and also store the reverse.

    config["metric"]:
        Cost metric for path length. Default: "euclidean".

    Returns
    -------
    dict
        Pairwise result dictionary.

        Keys:
            ("S", "G1"), ("G1", "S"), ...

        Values:
            standardized pairwise result dictionaries.
    """
    base_planner_class = config.get("base_planner_class")
    base_planner_config = config.get("base_planner_config")
    directed = config.get("directed", False)
    metric = config.get("metric", "euclidean")

    if base_planner_class is None:
        raise ValueError("Missing config['base_planner_class'].")

    if base_planner_config is None:
        raise ValueError("Missing config['base_planner_config'].")

    names = list(points.keys())
    pairwise_results: dict[tuple[str, str], dict[str, Any]] = {}

    if directed:
        pair_iterator = permutations(names, 2)
    else:
        pair_iterator = combinations(names, 2)

    for source_name, target_name in pair_iterator:
        result = plan_single_pair(
            base_planner_class=base_planner_class,
            collision_checker=collision_checker,
            source_config=points[source_name],
            target_config=points[target_name],
            base_planner_config=base_planner_config,
            metric=metric,
        )

        pairwise_results[(source_name, target_name)] = result

        if not directed:
            pairwise_results[(target_name, source_name)] = reverse_pairwise_result(result)

    return pairwise_results


def plan_single_pair(
    base_planner_class: type,
    collision_checker: Any,
    source_config: Any,
    target_config: Any,
    base_planner_config: dict[str, Any],
    metric: str = "euclidean",
) -> dict[str, Any]:
    """
    Plan one path between two configurations using one base planner call.

    Parameters
    ----------
    base_planner_class:
        Class of the base planner, e.g. BasicPRM or LazyPRM.

    collision_checker:
        Environment collision checker.

    source_config:
        Start configuration for this pair.

    target_config:
        Goal configuration for this pair.

    base_planner_config:
        Configuration passed to the base planner.

    metric:
        Metric used for final path length.

    Returns
    -------
    dict
        Standardized pairwise success or failure result.
    """
    planner = base_planner_class(collision_checker)

    start_time = time.perf_counter()

    try:
        path_nodes = planner.planPath(
            [source_config],
            [target_config],
            base_planner_config,
        )
    except Exception as exc:
        return pairwise_failure(
            metadata={
                "planner": base_planner_class.__name__,
                "exception": repr(exc),
                "planning_time": time.perf_counter() - start_time,
            }
        )

    planning_time = time.perf_counter() - start_time

    if path_nodes == []:
        return pairwise_failure(
            metadata={
                "planner": base_planner_class.__name__,
                "planning_time": planning_time,
                "reason": "Base planner returned no path.",
                **get_planner_graph_metadata(planner),
            }
        )

    try:
        path_configs = extract_path_configs(planner, path_nodes)
    except Exception as exc:
        return pairwise_failure(
            metadata={
                "planner": base_planner_class.__name__,
                "exception": repr(exc),
                "planning_time": planning_time,
                "reason": "Could not extract path configurations from planner graph.",
                **get_planner_graph_metadata(planner),
            }
        )

    cost = path_length(path_configs, metric=metric)

    return pairwise_success(
        path_nodes=path_nodes,
        path_configs=path_configs,
        cost=cost,
        metadata={
            "planner": base_planner_class.__name__,
            "planning_time": planning_time,
            "metric": metric,
            **get_planner_graph_metadata(planner),
        },
    )


def extract_path_configs(
    planner: Any,
    path_nodes: list[Any],
) -> list[Any]:
    """
    Convert base-planner path nodes into actual configurations.
    """
    path_configs = []

    for node in path_nodes:
        path_configs.append(planner.graph.nodes[node]["pos"])

    return path_configs


def reverse_pairwise_result(
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Reverse a pairwise result for undirected planning.

    If a collision-free path from A to B exists in a symmetric configuration
    space, then the same path reversed is a valid path from B to A.

    Failed results stay failed.
    """
    if not result.get("success", False):
        return pairwise_failure(
            metadata={
                **result.get("metadata", {}),
                "reversed": True,
            }
        )

    path_nodes = result.get("path_nodes")
    path_configs = result.get("path_configs")

    reversed_path_nodes = list(reversed(path_nodes)) if path_nodes is not None else None
    reversed_path_configs = list(reversed(path_configs)) if path_configs is not None else None

    return pairwise_success(
        path_nodes=reversed_path_nodes,
        path_configs=reversed_path_configs,
        cost=float(result["cost"]),
        metadata={
            **result.get("metadata", {}),
            "reversed": True,
        },
    )


def get_planner_graph_metadata(planner: Any) -> dict[str, Any]:
    """
    Extract basic graph statistics from a base planner.

    This is useful for benchmarking and debugging.
    """
    graph = getattr(planner, "graph", None)

    if graph is None:
        return {
            "roadmap_nodes": None,
            "roadmap_edges": None,
        }

    return {
        "roadmap_nodes": graph.number_of_nodes(),
        "roadmap_edges": graph.number_of_edges(),
    }