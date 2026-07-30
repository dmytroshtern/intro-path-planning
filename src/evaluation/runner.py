"""Run roundtrip benchmarks and collect the required metrics."""

import random
from contextlib import contextmanager
from itertools import product
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from src.multiquery.MultiQueryRoundtripPlanner import (
    MultiQueryRoundtripPlanner,
)
from src.roundtrip_algorithm import RoundtripPlanner

from .config import (
    MULTIQUERY_PLANNER_NAME,
    PLANAR_EXPERIMENT_SEED,
    build_roundtrip_config,
)


_COLLISION_METHODS = (
    "pointInCollision",
    "lineInCollision",
    "lineInCollisionExact",
)


@contextmanager
def _count_collision_calls(checker):
    """Count collision-check calls and restore the original methods."""
    counts = dict.fromkeys(_COLLISION_METHODS, 0)
    missing = object()
    originals = {}

    for name in _COLLISION_METHODS:
        if not hasattr(checker, name):
            continue
        method = getattr(checker, name)
        originals[name] = checker.__dict__.get(name, missing)

        def counted(*args, _name=name, _method=method, **kwargs):
            counts[_name] += 1
            return _method(*args, **kwargs)

        setattr(checker, name, counted)

    try:
        yield counts
    finally:
        for name, original in originals.items():
            if original is missing:
                delattr(checker, name)
            else:
                setattr(checker, name, original)


def _extract_metrics(
    result, benchmark, planner_name, order_method, seed,
    planning_time, experiment_id, collision_counts, error, planner_graph,
):
    """Convert one planner result to the report's metric schema."""
    result = result if isinstance(result, dict) else {}
    success = bool(result.get("success")) and not error
    path = result.get("final_path_configs") or []
    pairwise_results = [
        value
        for value in (result.get("pairwise_results") or {}).values()
        if not value.get("metadata", {}).get("reversed", False)
    ]
    successful = sum(bool(value.get("success")) for value in pairwise_results)
    failed = len(pairwise_results) - successful
    metadata = [value.get("metadata", {}) for value in pairwise_results]
    roadmap_nodes = sum(
        value.get("roadmap_nodes") or 0 for value in metadata
    )
    roadmap_edges = sum(
        value.get("roadmap_edges") or 0 for value in metadata
    )
    if planner_name == MULTIQUERY_PLANNER_NAME:
        successful = len(result.get("used_pairs") or []) if success else 0
        failed = 0
        if planner_graph is not None:
            roadmap_nodes = planner_graph.number_of_nodes()
            roadmap_edges = planner_graph.number_of_edges()
    tour_cost = result.get("tour_cost")

    return {
        "experiment_id": experiment_id,
        "benchmark": benchmark.name,
        "dof": benchmark.collisionChecker.getDim(),
        "number_of_goals": len(benchmark.goalList),
        "base_planner": planner_name,
        "order_method": order_method,
        "seed": seed,
        "success": success,
        "planning_time": planning_time,
        "final_path_length": float(tour_cost)
        if success and tour_cost is not None
        else np.nan,
        "final_path_points": len(path) if success else 0,
        "successful_subpaths": successful,
        "failed_subpaths": failed,
        "roadmap_nodes": int(roadmap_nodes),
        "roadmap_edges": int(roadmap_edges),
        "collision_checks": sum(collision_counts.values()),
        "error": error or result.get("reason", ""),
    }


class RoundtripBenchmarkRunner:
    """Run experiment combinations and retain results for later plots."""

    def __init__(self, seed=PLANAR_EXPERIMENT_SEED):
        self.records = {}
        self.seed = seed

    def clear(self):
        self.records.clear()

    def run_single(
        self,
        benchmark,
        base_planner_name,
        order_method,
        seed=None,
    ):
        seed = self.seed if seed is None else seed
        experiment_id = (
            f"{benchmark.name}|{base_planner_name}|{order_method}|seed_{seed}"
        )
        random.seed(seed)
        np.random.seed(seed)
        if base_planner_name == MULTIQUERY_PLANNER_NAME:
            planner = MultiQueryRoundtripPlanner(
                benchmark.collisionChecker,
            )
        else:
            planner = RoundtripPlanner(benchmark.collisionChecker)
        config = build_roundtrip_config(
            base_planner_name, order_method, seed, benchmark
        )
        result = None
        error = ""
        collision_counts = dict.fromkeys(_COLLISION_METHODS, 0)

        started = perf_counter()
        try:
            with _count_collision_calls(benchmark.collisionChecker) as collision_counts:
                result = planner.planPath(
                    benchmark.startList, benchmark.goalList, config
                )
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        planning_time = perf_counter() - started

        if not error and not isinstance(result, dict):
            error = "TypeError: RoundtripPlanner must return a dictionary."

        metrics = _extract_metrics(
            result,
            benchmark,
            base_planner_name,
            order_method,
            seed,
            planning_time,
            experiment_id,
            collision_counts,
            error,
            getattr(planner, "graph", None),
        )
        record = SimpleNamespace(
            result=result, benchmark=benchmark, metrics=metrics
        )
        self.records[experiment_id] = record
        return record

    def run_suite(
        self,
        benchmarks,
        base_planner_names,
        order_methods,
        seeds=None,
    ):
        records = []
        seeds = (self.seed,) if seeds is None else tuple(seeds)
        if not seeds:
            raise ValueError("At least one random seed is required.")

        combinations = product(
            benchmarks,
            base_planner_names,
            order_methods,
            seeds,
        )
        for benchmark, planner_name, order_method, seed in combinations:
            record = self.run_single(
                benchmark,
                planner_name,
                order_method,
                seed=seed,
            )
            records.append(record)
            status = "success" if record.metrics["success"] else "failed"
            print(record.metrics["experiment_id"], status)
        return records

    def find_result(
        self,
        benchmark_name,
        base_planner_name=None,
        order_method=None,
        seed=None,
        successful_only=True,
    ):
        for record in self.records.values():
            metrics = record.metrics
            matches = (
                metrics["benchmark"] == benchmark_name
                and (
                    base_planner_name is None
                    or metrics["base_planner"] == base_planner_name
                )
                and (
                    order_method is None
                    or metrics["order_method"] == order_method
                )
                and (
                    seed is None
                    or metrics["seed"] == seed
                )
                and (not successful_only or metrics["success"])
            )
            if matches:
                return record
        return None
