"""
Visit-order solvers for the Roundtrip Planner.

The ordering problem is solved on the metagraph.

Input:
    Metagraph with nodes "S", "G1", "G2", ...

Output:
    A visit order such as:
        ["S", "G2", "G1", "G3", "S"]

The exact solver tries all permutations.
The greedy solver repeatedly chooses the cheapest next unvisited goal.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any

import networkx as nx


def solve_order_exact(
    metagraph: nx.Graph | nx.DiGraph,
    start_name: str = "S",
) -> dict[str, Any]:
    """
    Find the cheapest valid roundtrip order by brute force.

    This is exact, but factorial in the number of goals.

    For small numbers of goals, this is fine and simple.

    Example
    -------
    Nodes:
        S, G1, G2, G3

    Tries:
        S -> G1 -> G2 -> G3 -> S
        S -> G1 -> G3 -> G2 -> S
        S -> G2 -> G1 -> G3 -> S
        ...

    Returns
    -------
    dict
        {
            "success": True/False,
            "order": ["S", "G2", "G1", "S"] or None,
            "cost": total_cost,
            "method": "exact",
        }
    """
    if start_name not in metagraph.nodes:
        return {
            "success": False,
            "order": None,
            "cost": float("inf"),
            "method": "exact",
            "reason": f"Start node {start_name} is not in the metagraph.",
        }

    goals = [node for node in metagraph.nodes if node != start_name]

    if len(goals) == 0:
        return {
            "success": False,
            "order": None,
            "cost": float("inf"),
            "method": "exact",
            "reason": "No goals in metagraph.",
        }

    best_order = None
    best_cost = float("inf")

    for goal_order in permutations(goals):
        candidate_order = [start_name, *goal_order, start_name]
        candidate_cost = compute_order_cost(metagraph, candidate_order)

        if candidate_cost < best_cost:
            best_cost = candidate_cost
            best_order = candidate_order

    if best_order is None or best_cost == float("inf"):
        return {
            "success": False,
            "order": None,
            "cost": float("inf"),
            "method": "exact",
            "reason": "No valid roundtrip order exists.",
        }

    return {
        "success": True,
        "order": best_order,
        "cost": best_cost,
        "method": "exact",
    }


def solve_order_greedy(
    metagraph: nx.Graph | nx.DiGraph,
    start_name: str = "S",
) -> dict[str, Any]:
    """
    Find a roundtrip order using a greedy nearest-neighbor heuristic.

    The algorithm repeatedly chooses the cheapest reachable unvisited goal.

    This is faster than exact search, but not guaranteed to be optimal.

    Returns
    -------
    dict
        {
            "success": True/False,
            "order": ["S", "G2", "G1", "S"] or None,
            "cost": total_cost,
            "method": "greedy",
        }
    """
    if start_name not in metagraph.nodes:
        return {
            "success": False,
            "order": None,
            "cost": float("inf"),
            "method": "greedy",
            "reason": f"Start node {start_name} is not in the metagraph.",
        }

    unvisited = set(node for node in metagraph.nodes if node != start_name)

    if len(unvisited) == 0:
        return {
            "success": False,
            "order": None,
            "cost": float("inf"),
            "method": "greedy",
            "reason": "No goals in metagraph.",
        }

    order = [start_name]
    current = start_name
    total_cost = 0.0

    while unvisited:
        next_node = None
        next_cost = float("inf")

        for candidate in unvisited:
            if not metagraph.has_edge(current, candidate):
                continue

            edge_cost = float(metagraph.edges[current, candidate]["weight"])

            if edge_cost < next_cost:
                next_cost = edge_cost
                next_node = candidate

        if next_node is None:
            return {
                "success": False,
                "order": None,
                "cost": float("inf"),
                "method": "greedy",
                "reason": f"No reachable unvisited goal from {current}.",
            }

        order.append(next_node)
        total_cost += next_cost
        unvisited.remove(next_node)
        current = next_node

    if not metagraph.has_edge(current, start_name):
        return {
            "success": False,
            "order": None,
            "cost": float("inf"),
            "method": "greedy",
            "reason": f"Cannot return from {current} to {start_name}.",
        }

    total_cost += float(metagraph.edges[current, start_name]["weight"])
    order.append(start_name)

    return {
        "success": True,
        "order": order,
        "cost": total_cost,
        "method": "greedy",
    }


def compute_order_cost(
    metagraph: nx.Graph | nx.DiGraph,
    order: list[str],
) -> float:
    """
    Compute the total cost of a complete visit order.

    If any required edge is missing, return infinity.

    Example
    -------
    order = ["S", "G1", "G2", "S"]

    cost =
        cost(S, G1)
        + cost(G1, G2)
        + cost(G2, S)
    """
    total = 0.0

    for source, target in zip(order, order[1:]):
        if not metagraph.has_edge(source, target):
            return float("inf")

        total += float(metagraph.edges[source, target]["weight"])

    return total


def is_valid_order(
    metagraph: nx.Graph | nx.DiGraph,
    order: list[str],
) -> bool:
    """
    Check whether all consecutive edges in the order exist.
    """
    return compute_order_cost(metagraph, order) < float("inf")