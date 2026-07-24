# coding: utf-8

"""Benchmark suites used for roundtrip evaluation and comparison."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from notebooks.IPBenchmark import Benchmark
from notebooks.IPEnvironment import CollisionChecker
from notebooks.IPEnvironmentKin import KinChainCollisionChecker
from notebooks.IPPlanarManipulator import PlanarRobot

from . import RoundtripTestSuite


def _lecture_benchmark(name):
    return next(
        benchmark
        for benchmark in RoundtripTestSuite.benchList
        if benchmark.name == name
    )


def _copy_point_environment(lecture_benchmark):
    checker = lecture_benchmark.collisionChecker
    return CollisionChecker(
        scene=dict(checker.scene),
        limits=checker.getEnvironmentLimits(),
    )


def create_point_robot_benchmarks(goal_counts=(3, 5, 8)):
    """Create goal-count variants for four 2-DoF point environments."""
    goal_counts = list(goal_counts)
    if not goal_counts:
        raise ValueError("At least one goal count is required.")
    if any(
        not isinstance(goal_count, int) or goal_count < 1
        for goal_count in goal_counts
    ):
        raise ValueError("Goal counts must be positive integers.")

    benchmark_specs = [
        {
            "lecture_name": "Trap",
            "name": "PointRobot Trap",
            "start": [10.0, 15.0],
            "goals": [
                [10.0, 1.0],
                [2.0, 20.0],
                [20.0, 20.0],
                [2.0, 2.0],
                [20.0, 2.0],
                [10.0, 20.0],
                [2.0, 10.0],
                [20.0, 10.0],
            ],
            "description": "Roundtrip from inside the trap.",
            "level": 1,
        },
        {
            "lecture_name": "Bottleneck",
            "name": "PointRobot Bottleneck",
            "start": [4.0, 15.0],
            "goals": [
                [18.0, 1.0],
                [4.0, 3.0],
                [18.0, 20.0],
                [8.0, 18.0],
                [3.0, 20.0],
                [20.0, 16.0],
                [8.0, 5.0],
                [15.0, 5.0],
            ],
            "description": "Roundtrip through one narrow passage.",
            "level": 2,
        },
        {
            "lecture_name": "Fat bottleneck",
            "name": "PointRobot Fat Bottleneck",
            "start": [4.0, 21.0],
            "goals": [
                [18.0, 1.0],
                [4.0, 3.0],
                [18.0, 20.0],
                [12.0, 4.0],
                [2.0, 20.0],
                [20.0, 3.0],
                [12.0, 18.0],
                [12.0, 11.0],
            ],
            "description": "Roundtrip through an extended narrow passage.",
            "level": 3,
        },
        {
            "lecture_name": "Alternating Gates",
            "name": "Alternating Gates Roundtrip",
            "start": [2.0, 7.0],
            "goals": [
                [8.0, 12.0],
                [13.0, 2.0],
                [18.0, 12.0],
                [23.0, 2.0],
                [28.0, 12.0],
                [4.0, 12.0],
                [8.0, 2.0],
                [28.0, 2.0],
            ],
            "description": (
                "Zig-zag roundtrip through alternating gates."
            ),
            "level": 3,
        },
    ]

    largest_goal_count = max(goal_counts)
    benchmarks = []
    for spec in benchmark_specs:
        if largest_goal_count > len(spec["goals"]):
            raise ValueError(
                f"{spec['name']} provides only {len(spec['goals'])} goals."
            )

        lecture_benchmark = _lecture_benchmark(spec["lecture_name"])
        for goal_count in goal_counts:
            name = spec["name"]
            if goal_count != goal_counts[0]:
                name += f" ({goal_count} goals)"

            benchmarks.append(
                Benchmark(
                    name,
                    _copy_point_environment(lecture_benchmark),
                    [spec["start"]],
                    spec["goals"][:goal_count],
                    (
                        f"{spec['description']} "
                        f"Number of goals: {goal_count}."
                    ),
                    spec["level"],
                )
            )

    return benchmarks


LECTURE_PLANAR_SCENE = {
    "obs1": LineString([(-2.0, 0.0), (-0.8, 0.0)]).buffer(0.5),
    "obs2": LineString([(2.0, 0.0), (2.0, 1.0)]).buffer(0.2),
    "obs3": LineString([(-1.0, 2.0), (1.0, 2.0)]).buffer(0.1),
}


def angular_distance(first, second):
    """Return wrapped Euclidean distance between joint configurations."""
    difference = np.asarray(first) - np.asarray(second)
    wrapped = (difference + np.pi) % (2.0 * np.pi) - np.pi
    return float(np.linalg.norm(wrapped))


def sample_free_configurations(
    environment,
    number_of_configurations,
    seed,
    minimum_distance=1.2,
    maximum_attempts=20_000,
):
    """Generate deterministic collision-free manipulator configurations."""
    generator = np.random.default_rng(seed)
    configurations = []

    for _ in range(maximum_attempts):
        candidate = generator.uniform(
            low=-np.pi,
            high=np.pi,
            size=environment.getDim(),
        ).tolist()

        if environment.pointInCollision(candidate):
            continue
        if any(
            angular_distance(candidate, existing) < minimum_distance
            for existing in configurations
        ):
            continue

        configurations.append(candidate)
        if len(configurations) == number_of_configurations:
            return configurations

    raise RuntimeError(
        f"Could not generate {number_of_configurations} free configurations."
    )


def create_planar_benchmark(
    name,
    degrees_of_freedom,
    scene,
    number_of_goals,
    seed,
    level,
    description,
    fk_resolution=0.15,
):
    """Create one reproducible PlanarManipulator roundtrip benchmark."""
    robot = PlanarRobot(n_joints=degrees_of_freedom)
    limits = [[-np.pi, np.pi] for _ in range(degrees_of_freedom)]
    environment = KinChainCollisionChecker(
        robot,
        scene=dict(scene),
        limits=limits,
        fk_resolution=fk_resolution,
    )
    configurations = sample_free_configurations(
        environment,
        number_of_configurations=number_of_goals + 1,
        seed=seed,
    )

    return Benchmark(
        name,
        environment,
        [configurations[0]],
        configurations[1:],
        description,
        level,
    )


def create_planar_robot_benchmarks():
    """Create the required 2-DoF and 4-DoF PlanarManipulator cases."""
    return [
        create_planar_benchmark(
            name="PlanarRobot 2-DoF Easy",
            degrees_of_freedom=2,
            scene={"obs2": LECTURE_PLANAR_SCENE["obs2"]},
            number_of_goals=3,
            seed=10,
            level=1,
            description="Two-joint robot with one workspace obstacle.",
        ),
        create_planar_benchmark(
            name="PlanarRobot 2-DoF Hard",
            degrees_of_freedom=2,
            scene=LECTURE_PLANAR_SCENE,
            number_of_goals=4,
            seed=21,
            level=3,
            description="Two-joint robot with the complete lecture scene.",
        ),
        create_planar_benchmark(
            name="PlanarRobot 4-DoF",
            degrees_of_freedom=4,
            scene={"obs2": LECTURE_PLANAR_SCENE["obs2"]},
            number_of_goals=2,
            seed=42,
            level=3,
            description=(
                "Four-joint robot with two goals in a "
                "four-dimensional configuration space and one obstacle."
            ),
            fk_resolution=0.35,
        ),
    ]


def validate_benchmark(benchmark):
    """Validate dimensions and collision freedom of required configurations."""
    environment = benchmark.collisionChecker
    configurations = benchmark.startList + benchmark.goalList

    for configuration in configurations:
        if len(configuration) != environment.getDim():
            raise ValueError(
                f"{benchmark.name}: configuration has the wrong dimension."
            )
        if environment.pointInCollision(configuration):
            raise ValueError(
                f"{benchmark.name}: required configuration is in collision."
            )

    return True


def benchmark_overview(benchmarks):
    """Return simple benchmark metadata for a notebook table."""
    rows = []
    for benchmark in benchmarks:
        robot_type = (
            "PlanarManipulator"
            if isinstance(benchmark.collisionChecker, KinChainCollisionChecker)
            else "PointRobot"
        )
        rows.append(
            {
                "benchmark": benchmark.name,
                "robot_type": robot_type,
                "dof": benchmark.collisionChecker.getDim(),
                "goals": len(benchmark.goalList),
                "difficulty": benchmark.level,
                "description": benchmark.description,
            }
        )
    return rows
