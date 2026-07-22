"""
Result helpers for the Roundtrip Planner.

This module defines small factory functions for standardized return
dictionaries. The goal is to keep pairwise planner results and final
roundtrip results consistent across the project.
"""

from __future__ import annotations

from typing import Any


def pairwise_success(
    path_nodes: list[Any] | None,
    path_configs: list[Any],
    cost: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized result for one successful pairwise path.
    """
    return {
        "success": True,
        "path_nodes": path_nodes,
        "path_configs": path_configs,
        "cost": float(cost),
        "metadata": metadata or {},
    }


def pairwise_failure(
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized result for one failed pairwise path.
    """
    return {
        "success": False,
        "path_nodes": None,
        "path_configs": None,
        "cost": float("inf"),
        "metadata": metadata or {},
    }


def roundtrip_success(
    visit_order: list[str],
    used_pairs: list[tuple[str, str]],
    final_path_configs: list[Any],
    tour_cost: float,
    pairwise_results: dict[tuple[str, str], dict[str, Any]],
    failed_pairs: list[tuple[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized successful roundtrip result.
    """
    return {
        "success": True,
        "visit_order": visit_order,
        "used_pairs": used_pairs,
        "final_path_configs": final_path_configs,
        "tour_cost": float(tour_cost),
        "pairwise_results": pairwise_results,
        "failed_pairs": failed_pairs or [],
        "metadata": metadata or {},
    }


def roundtrip_failure(
    reason: str,
    pairwise_results: dict[tuple[str, str], dict[str, Any]] | None = None,
    failed_pairs: list[tuple[str, str]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a standardized failed roundtrip result.
    """
    return {
        "success": False,
        "reason": reason,
        "visit_order": None,
        "used_pairs": [],
        "final_path_configs": None,
        "tour_cost": float("inf"),
        "pairwise_results": pairwise_results or {},
        "failed_pairs": failed_pairs or [],
        "metadata": metadata or {},
    }


def collect_failed_pairs(
    pairwise_results: dict[tuple[str, str], dict[str, Any]],
) -> list[tuple[str, str]]:
    """
    Extract all failed pairwise path keys from pairwise_results.
    """
    return [
        pair
        for pair, result in pairwise_results.items()
        if not result.get("success", False)
    ]