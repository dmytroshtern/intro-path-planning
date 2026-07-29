# coding: utf-8

"""Benchmark suites used for roundtrip evaluation and comparison."""

import numpy as np
from shapely.geometry import LineString

from notebooks.IPBenchmark import Benchmark
from notebooks.IPEnvironment import CollisionChecker
from notebooks.IPEnvironmentKin import KinChainCollisionChecker
from notebooks.IPPlanarManipulator import PlanarRobot

from . import RoundtripTestSuite


_LECTURE_BENCHMARKS = {
    benchmark.name: benchmark
    for benchmark in RoundtripTestSuite.benchList
}


class SelfCollisionKinChainChecker(KinChainCollisionChecker):
    """Check workspace obstacles and self-collisions of planar links."""

    SELF_COLLISION_TOLERANCE = 1e-6

    @classmethod
    def _links_self_collide(cls, joint_positions):
        links = [
            LineString([start, end])
            for start, end in zip(
                joint_positions[:-1],
                joint_positions[1:],
            )
        ]

        for first_index, first_link in enumerate(links):
            for second_index in range(first_index + 2, len(links)):
                distance = first_link.distance(links[second_index])
                if distance <= cls.SELF_COLLISION_TOLERANCE:
                    return True

        return False

    def pointInCollision(self, configuration):
        self.kin_chain.move(configuration)
        joint_positions = self.kin_chain.get_transforms()

        if self._links_self_collide(joint_positions):
            return True

        return any(
            self.segmentInCollision(start, end)
            for start, end in zip(
                joint_positions[:-1],
                joint_positions[1:],
            )
        )


# Lecture name, report name, description, difficulty, candidate goals
_POINT_ROBOT_SPECS = (
    (
        "Trap",
        "PointRobot Trap",
        "Roundtrip from inside the trap.",
        1,
        (
            (10.0, 1.0), (2.0, 20.0), (20.0, 20.0), (2.0, 2.0),
            (20.0, 2.0), (10.0, 20.0), (2.0, 10.0), (20.0, 10.0),
        ),
    ),
    (
        "Bottleneck",
        "PointRobot Bottleneck",
        "Roundtrip through one narrow passage.",
        2,
        (
            (18.0, 1.0), (4.0, 3.0), (18.0, 20.0), (8.0, 18.0),
            (3.0, 20.0), (20.0, 16.0), (8.0, 5.0), (15.0, 5.0),
        ),
    ),
    (
        "Fat bottleneck",
        "PointRobot Fat Bottleneck",
        "Roundtrip through an extended narrow passage.",
        3,
        (
            (18.0, 1.0), (4.0, 3.0), (18.0, 20.0), (12.0, 4.0),
            (2.0, 20.0), (20.0, 3.0), (12.0, 18.0), (12.0, 11.0),
        ),
    ),
    (
        "Alternating Gates",
        "Alternating Gates Roundtrip",
        "Zig-zag roundtrip through alternating gates.",
        3,
        (
            (8.0, 12.0), (13.0, 2.0), (18.0, 12.0), (23.0, 2.0),
            (28.0, 12.0), (4.0, 12.0), (8.0, 2.0), (28.0, 2.0),
        ),
    ),
)


def _copy_point_environment(lecture_benchmark):
    checker = lecture_benchmark.collisionChecker
    return CollisionChecker(
        scene=dict(checker.scene),
        limits=checker.getEnvironmentLimits(),
    )


def create_point_robot_benchmarks(
    goal_counts=(3, 5, 8),
    label_goal_count=False,
):
    """Create goal-count variants for four 2-DoF point environments."""
    goal_counts = tuple(goal_counts)
    if not goal_counts or any(
        not isinstance(goal_count, int) or goal_count < 1
        for goal_count in goal_counts
    ):
        raise ValueError("Goal counts must be positive integers.")

    largest_goal_count = max(goal_counts)
    benchmarks = []
    for lecture_name, name, description, level, goals in _POINT_ROBOT_SPECS:
        if largest_goal_count > len(goals):
            raise ValueError(
                f"{name} provides only {len(goals)} goals."
            )

        lecture_benchmark = _LECTURE_BENCHMARKS[lecture_name]
        for goal_count in goal_counts:
            benchmark_name = name
            if label_goal_count or goal_count != goal_counts[0]:
                benchmark_name += f" ({goal_count} goals)"

            benchmarks.append(
                Benchmark(
                    benchmark_name,
                    _copy_point_environment(lecture_benchmark),
                    [list(start) for start in lecture_benchmark.startList],
                    [list(goal) for goal in goals[:goal_count]],
                    f"{description} Number of goals: {goal_count}.",
                    level,
                )
            )

    return benchmarks


_PLANAR_SCENE = {
    "obs1": LineString([(-2.0, 0.0), (-0.8, 0.0)]).buffer(0.5),
    "obs2": LineString([(2.0, 0.0), (2.0, 1.0)]).buffer(0.2),
    "obs3": LineString([(-1.0, 2.0), (1.0, 2.0)]).buffer(0.1),
}


_SIX_DOF_SCENE = {
    "wall": LineString([(3.0, 1.0), (3.0, 4.0)]).buffer(0.3),
}


_PLANAR_EVALUATION_CASES = (
    (
        "PlanarRobot 2-DoF Easy",
        {"obs2": _PLANAR_SCENE["obs2"]},
        1,
        0.15,
        (
            (
                [2.787, -0.883],
                [[1.789, 0.574], [-1.292, 2.656], [-0.182, -2.948]],
            ),
            (
                [-0.689, -0.913],
                [[0.955, -0.961], [-2.795, -1.568], [2.142, 1.999]],
            ),
            (
                [-1.602, 0.424],
                [[0.112, 2.525], [-0.050, -1.703], [-2.556, -2.568]],
            ),
        ),
    ),
    (
        "PlanarRobot 2-DoF Hard",
        _PLANAR_SCENE,
        3,
        0.15,
        (
            (
                [-0.966, -0.354],
                [[1.984, 1.178], [0.399, 2.116], [-1.807, 2.604]],
            ),
            (
                [-1.160, 1.748],
                [[0.155, -0.949], [-2.184, -0.330], [-2.388, 1.266]],
            ),
            (
                [-2.058, -1.590],
                [[-0.420, 0.879], [0.780, -1.854], [-0.658, -2.171]],
            ),
        ),
    ),
    (
        "PlanarRobot 4-DoF",
        {"obs1": _PLANAR_SCENE["obs1"]},
        2,
        0.35,
        (
            (
                [-1.794, -0.523, 1.933, -1.420],
                [
                    [1.984, -2.465, -0.400, 2.129],
                    [-1.580, -1.481, 2.806, -1.826],
                    [-2.314, -0.078, -2.192, 0.820],
                ],
            ),
            (
                [1.129, -0.798, 1.348, 1.889],
                [
                    [0.448, -2.989, 1.556, -1.035],
                    [2.125, -0.252, -1.573, 0.427],
                    [1.763, 0.016, -0.306, -0.868],
                ],
            ),
            (
                [-1.651, -1.098, -0.313, -2.524],
                [
                    [-0.616, 0.193, -1.713, -0.632],
                    [0.073, 1.994, 1.353, -0.193],
                    [-1.210, -0.754, -2.087, 2.937],
                ],
            ),
        ),
    ),
)


def _create_planar_benchmark(
    name,
    scene,
    start_configuration,
    goal_configurations,
    level,
    description,
    fk_resolution=0.15,
):
    """Create one PlanarManipulator benchmark from fixed configurations."""
    degrees_of_freedom = len(start_configuration)
    robot = PlanarRobot(n_joints=degrees_of_freedom)
    limits = [[-np.pi, np.pi] for _ in range(degrees_of_freedom)]
    environment = SelfCollisionKinChainChecker(
        robot,
        scene=dict(scene),
        limits=limits,
        fk_resolution=fk_resolution,
    )

    return Benchmark(
        name,
        environment,
        [list(start_configuration)],
        [list(configuration) for configuration in goal_configurations],
        description,
        level,
    )


def create_planar_robot_benchmarks():
    """Create three configuration sets per evaluation case plus 6 DoF."""
    benchmarks = []
    for name, scene, level, resolution, variants in _PLANAR_EVALUATION_CASES:
        for index, (start, goals) in enumerate(variants, start=1):
            benchmarks.append(
                _create_planar_benchmark(
                    f"{name} (configuration {index})",
                    scene,
                    start,
                    goals,
                    level,
                    (
                        f"{name} with configuration set {index}: "
                        "one start and three goal configurations."
                    ),
                    fk_resolution=resolution,
                )
            )

    benchmarks.append(
        _create_planar_benchmark(
            "PlanarRobot 6-DoF",
            _SIX_DOF_SCENE,
            [1.458, 2.515, 1.566, 2.412, -1.712, 1.477],
            [
                [0.935, -1.364, -1.364, 2.135, -1.785, -0.732],
                [-0.529, -1.210, -0.915, 1.531, -1.066, 0.296],
                [-2.371, -0.468, 1.693, 2.109, -0.955, -1.534],
            ],
            3,
            "Optional six-joint robot with one workspace obstacle.",
            fk_resolution=0.5,
        )
    )
    return benchmarks


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
