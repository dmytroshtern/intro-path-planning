# coding: utf-8

"""Benchmark suites used for roundtrip evaluation and comparison."""

import numpy as np
from shapely.geometry import LineString, Point

from notebooks.IPBenchmark import Benchmark
from notebooks.IPEnvironment import CollisionChecker
from notebooks.IPEnvironmentKin import KinChainCollisionChecker
from notebooks.IPPlanarManipulator import PlanarRobot
from notebooks import IPTestSuite


_LECTURE_BENCHMARKS = {
    benchmark.name: benchmark
    for benchmark in IPTestSuite.benchList
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
    """Create goal-count variants for three lecture environments."""
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


# The 2- and 4-DoF cases use the obstacle scene from
# IP-10-1-PlanarManipulatorTests.ipynb.
_PLANAR_LECTURE_SCENE = {
    "obs1": LineString([(-2.0, 0.0), (-0.8, 0.0)]).buffer(0.5),
    "obs2": LineString([(2.0, 0.0), (2.0, 1.0)]).buffer(0.2),
    "obs3": LineString([(-1.0, 2.0), (1.0, 2.0)]).buffer(0.1),
}

_PLANAR_6_DOF_SCENE = {
    "obs1": LineString([(-2.1, 0.5), (-1.0, 0.5)]).buffer(0.3),
    "obs2": LineString([(1.45, -1.0), (1.45, 0.45)]).buffer(0.18),
    "obs3": LineString([(-0.5, 1.55), (0.9, 1.55)]).buffer(0.12),
    "obs4": Point(-0.4, -1.7).buffer(0.28),
}

_PLANAR_TOTAL_REACH = 3.0

_PLANAR_CASES = (
    (
        "PlanarRobot 2-DoF",
        _PLANAR_LECTURE_SCENE,
        "Lecture scene with two revolute joints.",
        1,
        (
            (
                [-1.0070, 1.3436],
                [
                    [-2.1370, -1.3293],
                    [-2.1565, 1.1714],
                    [0.1460, 1.5000],
                ],
            ),
            (
                [-0.9021, 0.7824],
                [
                    [-2.3642, 0.2711],
                    [1.9124, 1.1714],
                    [-1.3589, -1.9088],
                ],
            ),
        ),
    ),
    (
        "PlanarRobot 4-DoF",
        _PLANAR_LECTURE_SCENE,
        "Lecture scene with four shorter revolute links.",
        2,
        (
            (
                [1.2378, -0.5271, -0.4028, -0.5284],
                [
                    [1.8257, 1.0506, -0.3531, 0.7669],
                    [-2.1758, -0.4898, -0.1678, 0.7032],
                    [-0.8508, 0.1818, -0.3736, 0.8682],
                ],
            ),
            (
                [-1.0297, 1.6727, -0.7160, -0.2014],
                [
                    [2.1947, 0.0288, 0.8377, -0.1840],
                    [-2.1520, 0.5139, -1.0269, 0.8479],
                    [1.0536, -0.2315, 0.1746, 0.4703],
                ],
            ),
        ),
    ),
    (
        "PlanarRobot 6-DoF",
        _PLANAR_6_DOF_SCENE,
        "Additional obstacle and six short revolute links.",
        3,
        (
            (
                [0.6429, -0.1990, 0.2695, -0.9584, 0.2881, 0.7331],
                [
                    [0.4239, 0.0871, 0.5798, 0.1917, 0.5075, 0.6344],
                    [0.1394, 0.8415, 0.2582, -0.2990, -0.0409, 0.5457],
                    [0.3880, -0.2837, 0.9352, -0.3274, -0.0573, -0.0366],
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
    description,
    level,
):
    """Create one equally sized PlanarManipulator benchmark."""
    degrees_of_freedom = len(start_configuration)
    robot = PlanarRobot(n_joints=degrees_of_freedom)
    link_length = _PLANAR_TOTAL_REACH / degrees_of_freedom
    for joint in robot.joints:
        joint.a = link_length

    limits = [[-np.pi, np.pi] for _ in range(degrees_of_freedom)]
    fk_resolution = {
        2: 0.2,
        4: 0.35,
        6: 0.5,
    }[degrees_of_freedom]
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
        f"{description} Total reach: {_PLANAR_TOTAL_REACH:.1f}.",
        level,
    )


def create_planar_robot_benchmarks():
    """Create the selected PlanarManipulator configuration variants."""
    return [
        _create_planar_benchmark(
            f"{name} (configuration {variant_index})",
            scene,
            start,
            goals,
            f"{description} Configuration variant {variant_index}.",
            level,
        )
        for name, scene, description, level, variants in _PLANAR_CASES
        for variant_index, (start, goals) in enumerate(variants, start=1)
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
        is_planar = isinstance(
            benchmark.collisionChecker,
            KinChainCollisionChecker,
        )
        display_name = benchmark.name
        if display_name.endswith(" goals)"):
            display_name = display_name.rsplit(" (", 1)[0]

        row = {
            "benchmark": display_name,
            "robot_type": (
                "PlanarManipulator" if is_planar else "PointRobot"
            ),
            "goals": len(benchmark.goalList),
        }
        if not is_planar:
            row["difficulty"] = benchmark.level
        description = benchmark.description
        goal_suffix = (
            f" Number of goals: {len(benchmark.goalList)}."
        )
        if description.endswith(goal_suffix):
            description = description.removesuffix(goal_suffix)
        row["description"] = description
        rows.append(row)
    return rows
