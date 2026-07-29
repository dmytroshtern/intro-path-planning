"""Base-planner and experiment configurations for roundtrip evaluation."""

from copy import deepcopy

from notebooks.IPBasicPRM import BasicPRM
from notebooks.IPLazyPRM import LazyPRM
from notebooks.IPVisibilityPRM import VisPRM

from src.roundtrip_algorithm import compute_pairwise_paths


_PLANNER_CLASSES = {
    "BasicPRM": BasicPRM,
    "LazyPRM": LazyPRM,
    "VisibilityPRM": VisPRM,
}

_PLANNER_CONFIGS = {
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

def _planar_configs(radius):
    """Return the final planner settings for a planar configuration space."""
    return {
        "BasicPRM": {
            "radius": radius,
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


_TWO_DOF_PLANAR_CONFIGS = _planar_configs(radius=9.0)
_FOUR_DOF_CONFIGS = _planar_configs(radius=13.0)
_SIX_DOF_CONFIGS = _planar_configs(radius=16.0)

BASE_PLANNERS_TO_COMPARE = tuple(_PLANNER_CLASSES)
ORDER_METHODS_TO_COMPARE = ("exact", "greedy")
POINT_EXPERIMENT_SEEDS = (17, 31) #47) #, 73, 101)
PLANAR_EXPERIMENT_SEED = 17


def build_roundtrip_config(
    base_planner_name,
    order_method,
    seed,
    benchmark,
):
    """Build a configuration accepted by RoundtripPlanner."""
    if base_planner_name not in _PLANNER_CLASSES:
        raise KeyError(f"Unknown base planner: {base_planner_name}")
    if order_method not in ORDER_METHODS_TO_COMPARE:
        raise ValueError(
            f"Unknown order method: {order_method}. "
            f"Expected one of {ORDER_METHODS_TO_COMPARE}."
        )

    is_planar = hasattr(benchmark.collisionChecker, "kin_chain")
    if is_planar and benchmark.collisionChecker.getDim() == 6:
        planner_configs = _SIX_DOF_CONFIGS
    elif is_planar and benchmark.collisionChecker.getDim() == 4:
        planner_configs = _FOUR_DOF_CONFIGS
    elif is_planar:
        planner_configs = _TWO_DOF_PLANAR_CONFIGS
    else:
        planner_configs = _PLANNER_CONFIGS

    return {
        "pairwise_provider": compute_pairwise_paths,
        "base_planner_class": _PLANNER_CLASSES[base_planner_name],
        "base_planner_config": deepcopy(planner_configs[base_planner_name]),
        "ordering_method": order_method,
        "directed": False,
        "metric": "euclidean",
        "random_seed": seed,
    }
