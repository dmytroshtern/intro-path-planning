"""Base-planner and experiment configurations for roundtrip evaluation."""

from __future__ import annotations

from copy import deepcopy

from notebooks.IPBasicPRM import BasicPRM
from notebooks.IPLazyPRM import LazyPRM
from notebooks.IPVisibilityPRM import VisPRM

from src.roundtrip_algorithm import RoundtripPlanner, compute_pairwise_paths


BASE_PLANNER_CLASSES = {
    "BasicPRM": BasicPRM,
    "LazyPRM": LazyPRM,
    "VisibilityPRM": VisPRM,
}

BASE_PLANNER_CONFIGS = {
    "BasicPRM": {
        "radius": 5.0,
        "numNodes": 300,
        "collisionCheckingSteps": 40,
        "useKDTree": True,
    },
    "LazyPRM": {
        "initialRoadmapSize": 40,
        "updateRoadmapSize": 20,
        "kNearest": 5,
        "maxIterations": 40,
    },
    "VisibilityPRM": {
        "ntry": 40,
    },
}

FOUR_DOF_PLANAR_CONFIGS = {
    "BasicPRM": {
        "radius": 6.5,
        "numNodes": 60,
        "collisionCheckingSteps": 20,
        "useKDTree": True,
    },
    "LazyPRM": {
        "initialRoadmapSize": 15,
        "updateRoadmapSize": 10,
        "kNearest": 5,
        "maxIterations": 10,
    },
    "VisibilityPRM": {
        "ntry": 8,
    },
}

BASE_PLANNERS_TO_COMPARE = tuple(BASE_PLANNER_CLASSES)
ORDER_METHODS_TO_COMPARE = ("exact", "greedy")
DEVELOPMENT_SEEDS = (17,)
FINAL_EXPERIMENT_SEEDS = (11, 17, 23, 31, 43, 59, 71, 83, 97, 109)


def create_roundtrip_planner(benchmark):
    """Create a roundtrip planner for one benchmark environment."""
    return RoundtripPlanner(benchmark.collisionChecker)


def build_roundtrip_config(
    base_planner_name,
    order_method,
    seed,
    benchmark=None,
):
    """Build a configuration accepted by RoundtripPlanner."""
    if base_planner_name not in BASE_PLANNER_CLASSES:
        raise KeyError(f"Unknown base planner: {base_planner_name}")
    if order_method not in ORDER_METHODS_TO_COMPARE:
        raise ValueError(
            f"Unknown order method: {order_method}. "
            f"Expected one of {ORDER_METHODS_TO_COMPARE}."
        )

    is_four_dof_planar = (
        benchmark is not None
        and hasattr(benchmark.collisionChecker, "kin_chain")
        and benchmark.collisionChecker.getDim() == 4
    )
    planner_configs = (
        FOUR_DOF_PLANAR_CONFIGS
        if is_four_dof_planar
        else BASE_PLANNER_CONFIGS
    )

    return {
        "pairwise_provider": compute_pairwise_paths,
        "base_planner_class": BASE_PLANNER_CLASSES[base_planner_name],
        "base_planner_config": deepcopy(
            planner_configs[base_planner_name]
        ),
        "ordering_method": order_method,
        "directed": False,
        "metric": "euclidean",
        "random_seed": seed,
    }
