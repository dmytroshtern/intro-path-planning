"""Reusable benchmark runner for RoundtripPlanner."""

from __future__ import annotations

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from notebooks.IPEnvironmentKin import KinChainCollisionChecker

from .config import build_roundtrip_config, create_roundtrip_planner


REQUIRED_RESULT_KEYS = {
    "success",
    "visit_order",
    "used_pairs",
    "final_path_configs",
    "tour_cost",
    "pairwise_results",
    "failed_pairs",
    "metadata",
}


@dataclass
class ExperimentRecord:
    """Complete data from one planner run."""

    experiment_id: str
    planner: Any
    result: dict[str, Any] | None
    benchmark: Any
    config: dict[str, Any]
    metrics: dict[str, Any]


class CollisionCheckCounter:
    """Temporarily count collision-check calls on one environment."""

    METHOD_NAMES = (
        "pointInCollision",
        "lineInCollision",
        "lineInCollisionExact",
    )

    def __init__(self, collision_checker):
        self.collision_checker = collision_checker
        self.counts = {name: 0 for name in self.METHOD_NAMES}
        self._instance_values = {}

    def __enter__(self):
        instance_dictionary = getattr(self.collision_checker, "__dict__", {})
        for method_name in self.METHOD_NAMES:
            if not hasattr(self.collision_checker, method_name):
                continue

            original_method = getattr(self.collision_checker, method_name)
            self._instance_values[method_name] = instance_dictionary.get(
                method_name,
                None,
            )
            self._instance_values[(method_name, "had_value")] = (
                method_name in instance_dictionary
            )

            def counted_method(
                *args,
                _name=method_name,
                _method=original_method,
                **kwargs,
            ):
                self.counts[_name] += 1
                return _method(*args, **kwargs)

            setattr(self.collision_checker, method_name, counted_method)

        return self

    def __exit__(self, exception_type, exception, traceback):
        for method_name in self.METHOD_NAMES:
            marker = (method_name, "had_value")
            if marker not in self._instance_values:
                continue
            if self._instance_values[marker]:
                setattr(
                    self.collision_checker,
                    method_name,
                    self._instance_values[method_name],
                )
            else:
                delattr(self.collision_checker, method_name)


def validate_roundtrip_result(result, benchmark):
    """Validate the result contract and a successful final path."""
    if not isinstance(result, dict):
        raise TypeError("RoundtripPlanner.planPath must return a dictionary.")

    missing = REQUIRED_RESULT_KEYS.difference(result)
    if missing:
        raise KeyError(
            "Roundtrip result is missing keys: "
            + ", ".join(sorted(missing))
        )

    if not result["success"]:
        return True

    path = np.asarray(result["final_path_configs"], dtype=float)
    dimension = benchmark.collisionChecker.getDim()
    if path.ndim != 2 or path.shape[1] != dimension:
        raise ValueError("The final path has the wrong dimension.")
    if len(path) < 2:
        raise ValueError("A successful roundtrip needs at least two points.")

    start = np.asarray(benchmark.startList[0], dtype=float)
    if not np.allclose(path[0], start) or not np.allclose(path[-1], start):
        raise ValueError("The path must start and end at the benchmark start.")

    visit_order = result["visit_order"]
    expected_goals = {
        f"G{index}"
        for index in range(1, len(benchmark.goalList) + 1)
    }
    visited_goals = visit_order[1:-1]
    if (
        visit_order[0] != "S"
        or visit_order[-1] != "S"
        or set(visited_goals) != expected_goals
        or len(visited_goals) != len(expected_goals)
    ):
        raise ValueError("The visit order does not visit every goal once.")

    expected_pairs = list(zip(visit_order, visit_order[1:]))
    if result["used_pairs"] != expected_pairs:
        raise ValueError("used_pairs does not match visit_order.")

    for source, target in zip(path[:-1], path[1:]):
        if benchmark.collisionChecker.lineInCollision(source, target):
            raise ValueError("The final path contains a colliding segment.")

    return True


def original_pairwise_results(result):
    """Return actual planner calls without generated reverse copies."""
    return [
        pair_result
        for pair_result in result.get("pairwise_results", {}).values()
        if not pair_result.get("metadata", {}).get("reversed", False)
    ]


def extract_experiment_metrics(
    result,
    benchmark,
    base_planner_name,
    order_method,
    run_index,
    seed,
    planning_time,
    experiment_id,
    collision_counts,
    error="",
):
    """Translate one roundtrip result into the required metrics."""
    result = result or {}
    success = bool(result.get("success", False)) and not error
    path = result.get("final_path_configs") or []
    pairwise_results = original_pairwise_results(result)

    successful_subpaths = sum(
        bool(pair_result.get("success", False))
        for pair_result in pairwise_results
    )
    failed_subpaths = len(pairwise_results) - successful_subpaths
    roadmap_nodes = sum(
        pair_result.get("metadata", {}).get("roadmap_nodes") or 0
        for pair_result in pairwise_results
    )
    roadmap_edges = sum(
        pair_result.get("metadata", {}).get("roadmap_edges") or 0
        for pair_result in pairwise_results
    )

    point_checks = collision_counts.get("pointInCollision", 0)
    line_checks = collision_counts.get("lineInCollision", 0)
    exact_line_checks = collision_counts.get("lineInCollisionExact", 0)
    robot_type = (
        "PlanarManipulator"
        if isinstance(benchmark.collisionChecker, KinChainCollisionChecker)
        else "PointRobot"
    )

    return {
        "experiment_id": experiment_id,
        "benchmark": benchmark.name,
        "robot_type": robot_type,
        "dof": benchmark.collisionChecker.getDim(),
        "number_of_goals": len(benchmark.goalList),
        "difficulty": benchmark.level,
        "base_planner": base_planner_name,
        "order_method": order_method,
        "run": run_index,
        "seed": seed,
        "success": success,
        "planning_time": float(planning_time),
        "final_path_length": (
            float(result["tour_cost"])
            if success
            else np.nan
        ),
        "final_path_points": len(path) if success else 0,
        "successful_subpaths": successful_subpaths,
        "failed_subpaths": failed_subpaths,
        "roadmap_nodes": int(roadmap_nodes),
        "roadmap_edges": int(roadmap_edges),
        "point_collision_checks": int(point_checks),
        "line_collision_checks": int(line_checks),
        "exact_line_collision_checks": int(exact_line_checks),
        "collision_checks": int(
            point_checks + line_checks + exact_line_checks
        ),
        "error": error or result.get("reason", ""),
    }


class RoundtripBenchmarkRunner:
    """Execute experiment matrices and retain complete result records."""

    def __init__(self):
        self.records: dict[str, ExperimentRecord] = {}

    def clear(self):
        self.records.clear()

    def run_single(
        self,
        benchmark,
        base_planner_name,
        order_method,
        seed,
        run_index=0,
    ):
        experiment_id = (
            f"{benchmark.name}|{base_planner_name}|"
            f"{order_method}|run_{run_index:02d}|seed_{seed}"
        )
        random.seed(seed)
        np.random.seed(seed)
        planner = create_roundtrip_planner(benchmark)
        config = build_roundtrip_config(
            base_planner_name,
            order_method,
            seed,
            benchmark=benchmark,
        )
        result = None
        error = ""
        counter = CollisionCheckCounter(benchmark.collisionChecker)

        started = perf_counter()
        try:
            with counter:
                result = planner.planPath(
                    benchmark.startList,
                    benchmark.goalList,
                    config,
                )
            validate_roundtrip_result(result, benchmark)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
        planning_time = perf_counter() - started

        metrics = extract_experiment_metrics(
            result=result,
            benchmark=benchmark,
            base_planner_name=base_planner_name,
            order_method=order_method,
            run_index=run_index,
            seed=seed,
            planning_time=planning_time,
            experiment_id=experiment_id,
            collision_counts=counter.counts,
            error=error,
        )
        record = ExperimentRecord(
            experiment_id=experiment_id,
            planner=planner,
            result=result,
            benchmark=benchmark,
            config=config,
            metrics=metrics,
        )
        self.records[experiment_id] = record
        return record

    def run_suite(
        self,
        benchmarks: Iterable,
        base_planner_names: Iterable[str],
        order_methods: Iterable[str],
        seeds: Iterable[int],
    ):
        new_records = []
        for benchmark in benchmarks:
            for base_planner_name in base_planner_names:
                for order_method in order_methods:
                    for run_index, seed in enumerate(seeds):
                        record = self.run_single(
                            benchmark=benchmark,
                            base_planner_name=base_planner_name,
                            order_method=order_method,
                            seed=seed,
                            run_index=run_index,
                        )
                        new_records.append(record)
                        status = (
                            "success"
                            if record.metrics["success"]
                            else "failed"
                        )
                        print(record.experiment_id, status)
        return new_records

    def find_result(
        self,
        benchmark_name,
        base_planner_name=None,
        order_method=None,
        successful_only=True,
    ):
        for record in self.records.values():
            metrics = record.metrics
            if metrics["benchmark"] != benchmark_name:
                continue
            if (
                base_planner_name
                and metrics["base_planner"] != base_planner_name
            ):
                continue
            if order_method and metrics["order_method"] != order_method:
                continue
            if successful_only and not metrics["success"]:
                continue
            return record
        return None
