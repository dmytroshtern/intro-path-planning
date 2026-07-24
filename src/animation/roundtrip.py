"""Animations for PointRobot and PlanarManipulator roundtrip paths."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from notebooks.IPEnvironmentKin import (
    KinChainCollisionChecker,
    planarRobotVisualize,
)


def _require_success(result: dict[str, Any]) -> np.ndarray:
    if not result.get("success", False):
        reason = result.get("reason", "unknown reason")
        raise ValueError(f"Cannot animate a failed roundtrip: {reason}")

    path = np.asarray(result["final_path_configs"], dtype=float)
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("The final path must contain at least two configurations.")
    return path


def interpolate_path(path, frames_per_segment=10):
    """Linearly interpolate every path segment for smooth playback."""
    path = np.asarray(path, dtype=float)
    if frames_per_segment < 1:
        raise ValueError("frames_per_segment must be at least one.")

    interpolated = [path[0]]
    for start, target in zip(path[:-1], path[1:]):
        for fraction in np.linspace(
            0.0,
            1.0,
            frames_per_segment + 1,
        )[1:]:
            interpolated.append(start + fraction * (target - start))
    return np.asarray(interpolated)


def _draw_required_points(benchmark, ax) -> None:
    start = np.asarray(benchmark.startList[0], dtype=float)
    goals = np.asarray(benchmark.goalList, dtype=float)
    ax.scatter(
        start[0],
        start[1],
        marker="*",
        s=180,
        color="#2ca02c",
        edgecolor="black",
        label="Start",
        zorder=15,
    )
    ax.scatter(
        goals[:, 0],
        goals[:, 1],
        marker="X",
        s=90,
        color="#d62728",
        edgecolor="black",
        label="Goals",
        zorder=15,
    )


def animate_point_roundtrip(
    result,
    benchmark,
    frames_per_segment=10,
    interval=60,
):
    """Animate a 2-DoF PointRobot along the final roundtrip."""
    path = _require_success(result)
    if (
        benchmark.collisionChecker.getDim() != 2
        or isinstance(benchmark.collisionChecker, KinChainCollisionChecker)
    ):
        raise TypeError("A 2-DoF PointRobot benchmark is required.")

    frames = interpolate_path(path, frames_per_segment)
    limits = benchmark.collisionChecker.getEnvironmentLimits()
    figure, ax = plt.subplots(figsize=(8, 7))

    def update(frame_index):
        ax.clear()
        benchmark.collisionChecker.drawObstacles(ax)
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#b8c5d6",
            linewidth=2.0,
            label="Final path",
        )
        _draw_required_points(benchmark, ax)
        trace = frames[: frame_index + 1]
        ax.plot(
            trace[:, 0],
            trace[:, 1],
            color="#0057b8",
            linewidth=3.0,
            label="Travelled path",
        )
        ax.scatter(
            frames[frame_index, 0],
            frames[frame_index, 1],
            s=130,
            color="#ffbf00",
            edgecolor="black",
            zorder=20,
            label="PointRobot",
        )
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            f"{benchmark.name}: "
            + " → ".join(result["visit_order"])
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="best")

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=interval,
    )
    html = HTML(animation.to_jshtml())
    plt.close(figure)
    return html


def _workspace_limits(environment):
    robot = environment.kin_chain
    reach = sum(float(joint.a) for joint in robot.joints)
    obstacle_bounds = [
        geometry.bounds for geometry in environment.scene.values()
    ]
    if obstacle_bounds:
        reach = max(
            reach,
            max(abs(value) for bounds in obstacle_bounds for value in bounds),
        )
    margin = 0.5
    return [[-reach - margin, reach + margin]] * 2


def _draw_planar_robot(robot, ax):
    planarRobotVisualize(robot, ax)
    transforms = np.asarray(robot.get_transforms(), dtype=float)
    ax.scatter(
        transforms[:, 0],
        transforms[:, 1],
        s=40,
        color="#1f1f1f",
        zorder=10,
    )
    ax.scatter(
        transforms[-1, 0],
        transforms[-1, 1],
        s=80,
        marker="X",
        color="#ffbf00",
        edgecolor="black",
        zorder=11,
        label="End effector",
    )


def animate_planar_roundtrip(
    result,
    benchmark,
    workspace_limits=None,
    frames_per_segment=8,
    interval=60,
):
    """Animate a 2-DoF or 4-DoF PlanarManipulator in workspace."""
    path = _require_success(result)
    environment = benchmark.collisionChecker
    if not isinstance(environment, KinChainCollisionChecker):
        raise TypeError("A PlanarManipulator benchmark is required.")
    if path.shape[1] != environment.getDim():
        raise ValueError("Path dimension does not match the manipulator.")

    frames = interpolate_path(path, frames_per_segment)
    robot = deepcopy(environment.kin_chain)
    limits = workspace_limits or _workspace_limits(environment)
    show_configuration_space = environment.getDim() == 2

    if show_configuration_space:
        figure, (workspace_ax, configuration_ax) = plt.subplots(
            1,
            2,
            figsize=(14, 7),
        )
    else:
        figure, workspace_ax = plt.subplots(figsize=(8, 7))
        configuration_ax = None

    def update(frame_index):
        configuration = frames[frame_index]
        robot.move(configuration)

        workspace_ax.clear()
        environment.drawObstacles(workspace_ax, inWorkspace=True)
        _draw_planar_robot(robot, workspace_ax)
        workspace_ax.set_xlim(limits[0])
        workspace_ax.set_ylim(limits[1])
        workspace_ax.set_aspect("equal", adjustable="box")
        workspace_ax.set_xlabel("x")
        workspace_ax.set_ylabel("y")
        workspace_ax.set_title(
            f"{environment.getDim()}-DoF PlanarManipulator in workspace"
        )
        workspace_ax.grid(alpha=0.2)
        workspace_ax.legend(loc="upper right")

        if configuration_ax is not None:
            configuration_ax.clear()
            configuration_limits = environment.getEnvironmentLimits()
            configuration_ax.plot(
                path[:, 0],
                path[:, 1],
                color="#b8c5d6",
                linewidth=2.0,
                label="Final path",
            )
            trace = frames[: frame_index + 1]
            configuration_ax.plot(
                trace[:, 0],
                trace[:, 1],
                color="#0057b8",
                linewidth=3.0,
                label="Travelled path",
            )
            configuration_ax.scatter(
                configuration[0],
                configuration[1],
                s=110,
                color="#ffbf00",
                edgecolor="black",
                zorder=10,
                label="Current configuration",
            )
            _draw_required_points(benchmark, configuration_ax)
            configuration_ax.set_xlim(configuration_limits[0])
            configuration_ax.set_ylim(configuration_limits[1])
            configuration_ax.set_aspect("equal", adjustable="box")
            configuration_ax.set_xlabel(r"$q_1$ [rad]")
            configuration_ax.set_ylabel(r"$q_2$ [rad]")
            configuration_ax.set_title("Configuration space")
            configuration_ax.grid(alpha=0.2)
            configuration_ax.legend(loc="best", fontsize=8)

        figure.suptitle(
            f"{benchmark.name}: "
            + " → ".join(result["visit_order"])
        )

    animation = FuncAnimation(
        figure,
        update,
        frames=len(frames),
        interval=interval,
    )
    html = HTML(animation.to_jshtml())
    plt.close(figure)
    return html
