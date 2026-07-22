"""
Main Roundtrip Planner.

This module implements the high-level roundtrip planning pipeline:

1. Convert startList and goalList into named configurations.
2. Compute or receive pairwise paths between all required configurations.
3. Build a metagraph from successful pairwise paths.
4. Determine a visit order on the metagraph.
5. Concatenate the selected pairwise paths into one final roundtrip path.

The actual low-level path planning is delegated to existing lecture planners
such as BasicPRM, LazyPRM, or VisibilityPRM.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .metagraph import build_metagraph
from .ordering import solve_order_exact, solve_order_greedy
from .result import (
    collect_failed_pairs,
    roundtrip_failure,
    roundtrip_success,
)


class RoundtripPlanner:
    """
    High-level planner for roundtrip paths.

    The planner follows the same general interface style as the lecture
    planners:

        planPath(startList, goalList, config)

    Difference to normal planners:
        - startList contains one start configuration.
        - goalList contains multiple target/intermediate configurations.
        - The returned result is a dictionary containing the final path and
          metadata, not only a raw node list.

    The RoundtripPlanner itself does not implement PRM. It receives pairwise
    path results from a pairwise planning function/provider.
    """

    def __init__(self, collision_checker: Any | None = None):
        """
        Parameters
        ----------
        collision_checker:
            Collision checker/environment used by the base planner. This can
            be None when using precomputed or mock pairwise results.
        """
        self.collision_checker = collision_checker
        self.metagraph = None
        self.pairwise_results = None
        self.last_result = None

    def planPath(
        self,
        startList: list[Any],
        goalList: list[Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Plan a collision-free roundtrip path.
        """
        start_time = time.perf_counter()

        try:
            points = make_points(startList, goalList)
        except ValueError as exc:
            result = roundtrip_failure(
                reason=str(exc),
                metadata={"stage": "input_validation"},
            )
            self.last_result = result
            return result

        directed = config.get("directed", False)
        ordering_method = config.get("ordering_method", "exact")

        pairwise_results = self._get_pairwise_results(points, config)
        self.pairwise_results = pairwise_results

        failed_pairs = collect_failed_pairs(pairwise_results)

        metagraph = build_metagraph(
            points=points,
            pairwise_results=pairwise_results,
            directed=directed,
        )
        self.metagraph = metagraph

        if metagraph.number_of_edges() == 0:
            result = roundtrip_failure(
                reason="No successful pairwise paths found; metagraph has no edges.",
                pairwise_results=pairwise_results,
                failed_pairs=failed_pairs,
                metadata={
                    "stage": "metagraph",
                    "ordering_method": ordering_method,
                    "total_planning_time": time.perf_counter() - start_time,
                },
            )
            self.last_result = result
            return result

        order_result = solve_visit_order(
            metagraph=metagraph,
            start_name="S",
            method=ordering_method,
        )

        if not order_result["success"]:
            result = roundtrip_failure(
                reason="No valid roundtrip order found on the metagraph.",
                pairwise_results=pairwise_results,
                failed_pairs=failed_pairs,
                metadata={
                    "stage": "ordering",
                    "ordering_method": ordering_method,
                    "metagraph_nodes": metagraph.number_of_nodes(),
                    "metagraph_edges": metagraph.number_of_edges(),
                    "total_planning_time": time.perf_counter() - start_time,
                },
            )
            self.last_result = result
            return result

        visit_order = order_result["order"]
        used_pairs = make_used_pairs(visit_order)

        try:
            final_path_configs = concatenate_order_paths(
                visit_order=visit_order,
                pairwise_results=pairwise_results,
            )
        except KeyError as exc:
            result = roundtrip_failure(
                reason=f"Missing pairwise result while concatenating paths: {exc}",
                pairwise_results=pairwise_results,
                failed_pairs=failed_pairs,
                metadata={
                    "stage": "concatenation",
                    "ordering_method": ordering_method,
                    "total_planning_time": time.perf_counter() - start_time,
                },
            )
            self.last_result = result
            return result

        metadata = {
            "ordering_method": ordering_method,
            "directed": directed,
            "num_required_points": len(points),
            "num_goals": len(goalList),
            "num_pairwise_results": len(pairwise_results),
            "num_pairwise_success": sum(
                1 for result in pairwise_results.values() if result.get("success", False)
            ),
            "num_pairwise_failed": len(failed_pairs),
            "metagraph_nodes": metagraph.number_of_nodes(),
            "metagraph_edges": metagraph.number_of_edges(),
            "total_planning_time": time.perf_counter() - start_time,
        }

        if "base_planner" in config:
            metadata["base_planner"] = config["base_planner"]

        result = roundtrip_success(
            visit_order=visit_order,
            used_pairs=used_pairs,
            final_path_configs=final_path_configs,
            tour_cost=order_result["cost"],
            pairwise_results=pairwise_results,
            failed_pairs=failed_pairs,
            metadata=metadata,
        )

        self.last_result = result
        return result

    def _get_pairwise_results(
        self,
        points: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[tuple[str, str], dict[str, Any]]:
        """
        Get pairwise path results.

        There are two supported modes:

        1. Precomputed pairwise results:
            config["pairwise_results"]

        2. Pairwise provider function:
            config["pairwise_provider"]

        This design allows the same RoundtripPlanner to work with:
            - mock data,
            - repeated BasicPRM/LazyPRM/VisibilityPRM calls,
            - multi-query VisibilityPRM roadmap reuse.
        """
        if "pairwise_results" in config:
            return config["pairwise_results"]

        pairwise_provider: Callable | None = config.get("pairwise_provider")

        if pairwise_provider is None:
            raise ValueError(
                "No pairwise path source provided. "
                "Pass either config['pairwise_results'] for testing or "
                "config['pairwise_provider'] for real planner calls."
            )

        return pairwise_provider(
            points=points,
            collision_checker=self.collision_checker,
            config=config,
        )


def make_points(
    startList: list[Any],
    goalList: list[Any],
) -> dict[str, Any]:
    """
    Convert startList and goalList into named required configurations.
    """
    if startList is None or len(startList) != 1:
        raise ValueError(
            "RoundtripPlanner expects exactly one start configuration in startList."
        )

    if goalList is None or len(goalList) == 0:
        raise ValueError(
            "RoundtripPlanner expects at least one goal configuration in goalList."
        )

    points = {"S": startList[0]}

    for index, goal in enumerate(goalList, start=1):
        points[f"G{index}"] = goal

    return points


def solve_visit_order(
    metagraph: Any,
    start_name: str = "S",
    method: str = "exact",
) -> dict[str, Any]:
    """
    Dispatch to the selected visit-order strategy.

    Supported methods:
        - "exact"
        - "greedy"
    """
    if method == "exact":
        return solve_order_exact(metagraph, start_name=start_name)

    if method == "greedy":
        return solve_order_greedy(metagraph, start_name=start_name)

    raise ValueError(f"Unknown ordering method: {method}")


def make_used_pairs(visit_order: list[str]) -> list[tuple[str, str]]:
    """
    Convert a visit order into consecutive pair keys.

    Example
    -------
    ["S", "G2", "G1", "S"]

    becomes:

    [("S", "G2"), ("G2", "G1"), ("G1", "S")]
    """
    return list(zip(visit_order, visit_order[1:]))


def concatenate_order_paths(
    visit_order: list[str],
    pairwise_results: dict[tuple[str, str], dict[str, Any]],
) -> list[Any]:
    """
    Concatenate the pairwise subpaths selected by the visit order.

    Duplicate configurations at joins are removed.

    Example
    -------
    path(S, G1)  = [S, a, G1]
    path(G1, G2) = [G1, b, G2]

    final path:
        [S, a, G1, b, G2]
    """
    final_path = []

    for pair_index, pair in enumerate(make_used_pairs(visit_order)):
        if pair not in pairwise_results:
            raise KeyError(pair)

        pair_result = pairwise_results[pair]

        if not pair_result.get("success", False):
            raise KeyError(f"Pair exists but was not successful: {pair}")

        subpath = pair_result.get("path_configs")

        if subpath is None or len(subpath) == 0:
            raise KeyError(f"Pair has no path_configs: {pair}")

        if pair_index == 0:
            final_path.extend(subpath)
        else:
            final_path.extend(subpath[1:])

    return final_path