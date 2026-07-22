"""
Metagraph construction for the Roundtrip Planner.

The metagraph is the high-level graph used to decide the visit order.

Nodes:
    Required configurations only:
    "S", "G1", "G2", ...

Edges:
    Successful pairwise paths between required configurations.

Each edge stores:
    - weight: numeric path cost
    - path_nodes: internal base-planner node path
    - path_configs: actual configurations along the path
    - metadata: optional planner information
"""

from __future__ import annotations

from typing import Any

import networkx as nx


def build_metagraph(
    points: dict[str, Any],
    pairwise_results: dict[tuple[str, str], dict[str, Any]],
    directed: bool = False,
) -> nx.Graph | nx.DiGraph:
    """
    Build a metagraph from pairwise planning results.

    Parameters
    ----------
    points:
        Named required configurations.

        Example:
            {
                "S": [1, 1],
                "G1": [8, 2],
                "G2": [8, 8],
            }

    pairwise_results:
        Dictionary mapping pairs to pairwise result dictionaries.

        Example:
            {
                ("S", "G1"): {
                    "success": True,
                    "path_nodes": ["start", 3, 7, "goal"],
                    "path_configs": [[1, 1], [3, 2], [8, 2]],
                    "cost": 8.5,
                    "metadata": {},
                }
            }

    directed:
        If True, build a directed graph.
        If False, build an undirected graph.

    Returns
    -------
    nx.Graph or nx.DiGraph
        Metagraph containing only successful pairwise paths as edges.
    """
    graph = nx.DiGraph() if directed else nx.Graph()

    for name, config in points.items():
        graph.add_node(name, config=config)

    for (source, target), result in pairwise_results.items():
        if not result.get("success", False):
            continue

        if source not in points or target not in points:
            raise ValueError(
                f"Pair ({source}, {target}) contains unknown point name."
            )

        graph.add_edge(
            source,
            target,
            weight=float(result["cost"]),
            path_nodes=result.get("path_nodes"),
            path_configs=result.get("path_configs"),
            metadata=result.get("metadata", {}),
        )

    return graph


def edge_exists(
    metagraph: nx.Graph | nx.DiGraph,
    source: str,
    target: str,
) -> bool:
    """
    Check whether the metagraph contains an edge.
    """
    return metagraph.has_edge(source, target)


def get_edge_cost(
    metagraph: nx.Graph | nx.DiGraph,
    source: str,
    target: str,
) -> float:
    """
    Return the cost of one metagraph edge.
    """
    if not metagraph.has_edge(source, target):
        return float("inf")

    return float(metagraph.edges[source, target]["weight"])


def get_edge_path_configs(
    metagraph: nx.Graph | nx.DiGraph,
    source: str,
    target: str,
) -> list[Any]:
    """
    Return the stored path configurations of one metagraph edge.
    """
    if not metagraph.has_edge(source, target):
        raise KeyError(f"No edge exists between {source} and {target}.")

    path_configs = metagraph.edges[source, target].get("path_configs")

    if path_configs is None:
        raise KeyError(f"Edge ({source}, {target}) has no path_configs.")

    return path_configs