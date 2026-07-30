"""Roundtrip animations based on the lecture animation utilities."""

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from notebooks.IPEnvironmentKin import (
    KinChainCollisionChecker,
    interpolate_line,
    planarRobotVisualize,
)


def _result_path(result):
    if not result.get("success"):
        raise ValueError(
            "Cannot animate a failed roundtrip: "
            + result.get("reason", "unknown reason")
        )
    path = np.asarray(result["final_path_configs"], dtype=float)
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("The final path needs at least two configurations.")
    return path


def _interpolate_path(path, frames_per_segment):
    """Apply the lecture interpolation to every roundtrip segment."""
    if frames_per_segment < 1:
        raise ValueError("frames_per_segment must be at least one.")

    frames = [path[0]]
    for start, target in zip(path[:-1], path[1:]):
        distance = np.linalg.norm(target - start)
        if np.isclose(distance, 0.0):
            continue

        segment = interpolate_line(
            start,
            target,
            distance / frames_per_segment,
        )
        if not np.allclose(segment[-1], target):
            segment.append(target)
        frames.extend(segment[1:])
    return np.asarray(frames)


def _draw_start_and_goals(benchmark, ax):
    start = np.asarray(benchmark.startList[0])
    goals = np.asarray(benchmark.goalList)
    ax.scatter(
        start[0],
        start[1],
        marker="*",
        s=180,
        color="#2ca02c",
        edgecolor="black",
        label="Start",
        zorder=10,
    )
    ax.annotate(
        "S",
        start,
        xytext=(6, 6),
        textcoords="offset points",
        color="#2ca02c",
        fontweight="bold",
    )
    ax.scatter(
        goals[:, 0],
        goals[:, 1],
        marker="X",
        s=90,
        color="#d62728",
        edgecolor="black",
        label="Goals",
        zorder=10,
    )
    for index, goal in enumerate(goals, start=1):
        ax.annotate(
            f"G{index}",
            goal,
            xytext=(6, 6),
            textcoords="offset points",
            color="#d62728",
            fontweight="bold",
        )


def _configuration_collision_map(environment, samples=80):
    """Sample the 2-DoF collision regions as shown in the lecture."""
    limits = np.asarray(environment.getEnvironmentLimits(), dtype=float)
    q1_values = np.linspace(*limits[0], samples)
    q2_values = np.linspace(*limits[1], samples)
    collision_map = np.zeros((samples, samples), dtype=bool)
    original_configuration = [
        joint.theta for joint in environment.kin_chain.joints
    ]

    for row, q2 in enumerate(q2_values):
        for column, q1 in enumerate(q1_values):
            collision_map[row, column] = environment.pointInCollision(
                [q1, q2]
            )

    environment.kin_chain.move(original_configuration)
    return q1_values, q2_values, collision_map


def _draw_configuration_obstacles(ax, collision_data):
    q1_values, q2_values, collision_map = collision_data
    ax.contourf(
        q1_values,
        q2_values,
        collision_map,
        levels=[0.5, 1.5],
        colors=["#ff9999"],
        alpha=0.45,
    )
    ax.plot(
        [],
        [],
        color="#ff9999",
        linewidth=8,
        alpha=0.45,
        label="Collision region",
    )


def _draw_path_progress(ax, path, frames, frame_index, marker_label):
    ax.plot(
        path[:, 0],
        path[:, 1],
        color="#b8c5d6",
        linewidth=2,
        label="Final path",
    )
    trace = frames[: frame_index + 1]
    ax.plot(
        trace[:, 0],
        trace[:, 1],
        color="#0057b8",
        linewidth=3,
        label="Travelled path",
    )
    ax.scatter(
        *frames[frame_index],
        s=120,
        color="#ffbf00",
        edgecolor="black",
        label=marker_label,
        zorder=15,
    )


def _configure_axis(ax, limits, xlabel, ylabel):
    ax.set_xlim(limits[0])
    ax.set_ylim(limits[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)


def _workspace_references(robot, benchmark):
    """Precompute the workspace poses of start and goal configurations."""
    reference_robot = deepcopy(robot)
    required = [("S", benchmark.startList[0])]
    required.extend(
        (f"G{index}", goal)
        for index, goal in enumerate(benchmark.goalList, start=1)
    )

    poses = []
    for label, configuration in required:
        reference_robot.move(configuration)
        poses.append(
            (
                label,
                np.asarray(reference_robot.get_transforms()),
            )
        )

    return poses


def _draw_workspace_references(ax, poses, show_robot_poses):
    """Draw start and goals in the workspace."""
    for index, (label, points) in enumerate(poses):
        is_start = label == "S"
        color = "#2ca02c" if is_start else "#d62728"
        if show_robot_poses:
            ax.plot(
                points[:, 0],
                points[:, 1],
                color=color,
                linestyle="--",
                linewidth=1.5,
                alpha=0.3,
            )
        end_effector = points[-1]
        ax.scatter(
            *end_effector,
            color=color,
            marker="*" if is_start else "X",
            edgecolor="black",
            s=120 if is_start else 75,
            alpha=0.9,
            zorder=8,
            label=(
                "Start"
                if is_start
                else "Goals" if index == 1 else None
            ),
        )
        ax.annotate(
            label,
            end_effector,
            xytext=(6, 6),
            textcoords="offset points",
            color=color,
            fontweight="bold",
        )

def _animation_html(figure, update, number_of_frames, interval):
    animation = FuncAnimation(
        figure,
        update,
        frames=number_of_frames,
        interval=interval,
    )
    html = HTML(animation.to_jshtml())
    plt.close(figure)
    return html


def animate_point_roundtrip(
    result,
    benchmark,
    frames_per_segment=10,
    interval=60,
):
    """Animate a 2-DoF PointRobot in its environment."""
    path = _result_path(result)
    environment = benchmark.collisionChecker
    if (
        environment.getDim() != 2
        or isinstance(environment, KinChainCollisionChecker)
    ):
        raise TypeError("A 2-DoF PointRobot benchmark is required.")

    frames = _interpolate_path(path, frames_per_segment)
    limits = environment.getEnvironmentLimits()
    figure, ax = plt.subplots(figsize=(8, 7))
    title = f"{benchmark.name}: {' → '.join(result['visit_order'])}"

    def update(frame_index):
        ax.clear()
        environment.drawObstacles(ax)
        _draw_path_progress(
            ax,
            path,
            frames,
            frame_index,
            marker_label="PointRobot",
        )
        _draw_start_and_goals(benchmark, ax)
        _configure_axis(ax, limits, "x", "y")
        ax.set_title(title)
        ax.legend(loc="best")

    return _animation_html(figure, update, len(frames), interval)


def animate_planar_roundtrip(
    result,
    benchmark,
    workspace_limits=None,
    frames_per_segment=8,
    interval=60,
):
    """Animate a PlanarRobot roundtrip in the workspace."""
    path = _result_path(result)
    environment = benchmark.collisionChecker
    if not isinstance(environment, KinChainCollisionChecker):
        raise TypeError("A PlanarManipulator benchmark is required.")

    frames = _interpolate_path(path, frames_per_segment)
    robot = deepcopy(environment.kin_chain)
    dof = environment.getDim()
    if workspace_limits is None:
        reach = sum(joint.a for joint in robot.joints) + 0.5
        workspace_limits = [[-reach, reach], [-reach, reach]]
    workspace_poses = _workspace_references(robot, benchmark)
    if dof == 2:
        configuration_limits = environment.getEnvironmentLimits()
        collision_data = _configuration_collision_map(environment)
        figure, (workspace_ax, configuration_ax) = plt.subplots(
            1, 2, figsize=(14, 7)
        )
    else:
        configuration_ax = None
        figure, workspace_ax = plt.subplots(figsize=(8, 7))

    title = f"{benchmark.name}: {' → '.join(result['visit_order'])}"

    def update(frame_index):
        configuration = frames[frame_index]
        robot.move(configuration)

        workspace_ax.clear()
        environment.drawObstacles(workspace_ax, inWorkspace=True)
        _draw_workspace_references(
            workspace_ax,
            workspace_poses,
            show_robot_poses=dof > 2,
        )
        planarRobotVisualize(robot, workspace_ax)
        workspace_ax.plot(
            [],
            [],
            color="green",
            linewidth=3,
            label="Current robot",
        )
        _configure_axis(workspace_ax, workspace_limits, "x", "y")
        workspace_ax.set_title(
            f"{dof}-DoF PlanarManipulator in workspace"
        )
        workspace_ax.legend(loc="upper right", fontsize=8)

        if configuration_ax is not None:
            configuration_ax.clear()
            _draw_configuration_obstacles(
                configuration_ax,
                collision_data,
            )
            _draw_path_progress(
                configuration_ax,
                path,
                frames,
                frame_index,
                marker_label="Current configuration",
            )
            _draw_start_and_goals(benchmark, configuration_ax)
            _configure_axis(
                configuration_ax,
                configuration_limits,
                r"$q_1$ [rad]",
                r"$q_2$ [rad]",
            )
            configuration_ax.set_title("Configuration space")
            configuration_ax.legend(loc="best", fontsize=8)

        figure.suptitle(title)

    return _animation_html(figure, update, len(frames), interval)
