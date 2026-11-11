import argparse
import time
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from lab_common import artifact_dir, load_lab12_model, open_lab_viewer, validation_mode, write_csv

from controllers.navigation import SmartNavigator

if TYPE_CHECKING:
    from pathlib import Path

GOAL = np.array([6.0, 0.0], dtype=float)
MAX_STEPS = 60000
SAMPLE_EVERY = 50
SNAPSHOT_INTERVAL_S = 5.0
SNAPSHOT_FINAL_TIME_S = 15.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab 12: autonomous navigation with A*, DWA and live map.")
    parser.add_argument("--no-viz", action="store_true", help="Disable live map visualization.")
    parser.add_argument("--save-snapshots", action="store_true", help="Save map snapshots every 5 simulated seconds.")
    return parser.parse_args()


def setup_artifacts() -> Path:
    return artifact_dir("lab12")


def save_trajectory(positions: list[dict[str, object]], filename: Path) -> None:
    write_csv(filename, positions)


def save_occupancy_grid(grid: np.ndarray, filename: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.T, origin="lower", cmap="Greys", vmin=0, vmax=2, interpolation="nearest")
    plt.title("Lab12 final occupancy grid")
    plt.xlabel("grid x")
    plt.ylabel("grid y")
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close()


def save_path_visualization(path: list[np.ndarray], obstacles: list[tuple[float, float]], filename: Path) -> None:
    plt.figure(figsize=(7, 4))
    if path:
        path_array = np.asarray(path)
        plt.plot(path_array[:, 0], path_array[:, 1], "r--", label="A* path")
    if obstacles:
        obstacle_array = np.asarray(obstacles)
        plt.plot(obstacle_array[:, 0], obstacle_array[:, 1], "ks", markersize=3, label="known obstacles")
    plt.plot(0.0, 0.0, "go", label="start")
    plt.plot(GOAL[0], GOAL[1], "rx", markersize=10, label="goal")
    plt.axis("equal")
    plt.grid(visible=True, alpha=0.3)
    plt.legend(loc="best")
    plt.title("Lab12 initial A* plan")
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close()


def save_map_snapshot(
    navigator: SmartNavigator,
    robot_pos: np.ndarray,
    path: list[np.ndarray],
    goal: np.ndarray,
    trajectory: list[np.ndarray],
    filename: Path,
) -> None:
    extent = (
        navigator.grid.origin[0],
        navigator.grid.origin[0] + navigator.grid.size[0] * navigator.grid.resolution,
        navigator.grid.origin[1],
        navigator.grid.origin[1] + navigator.grid.size[1] * navigator.grid.resolution,
    )
    plt.figure(figsize=(7, 6))
    plt.imshow(
        navigator.grid.grid.T,
        origin="lower",
        extent=extent,
        cmap="Greys",
        vmin=0,
        vmax=2,
        interpolation="nearest",
    )
    if path:
        path_array = np.asarray(path)
        plt.plot(path_array[:, 0], path_array[:, 1], "r--", linewidth=1.2, label="A* path")
    if trajectory:
        trajectory_array = np.asarray(trajectory)
        plt.plot(trajectory_array[:, 0], trajectory_array[:, 1], "c-", linewidth=1.5, label="trajectory")
    plt.plot(robot_pos[0], robot_pos[1], "bo", label="robot")
    plt.plot(goal[0], goal[1], "gx", markersize=10, label="goal")
    plt.axis("equal")
    plt.grid(visible=True, alpha=0.3)
    plt.legend(loc="best")
    plt.title("Lab12 occupancy map snapshot")
    plt.tight_layout()
    plt.savefig(filename, dpi=140)
    plt.close()


def run_navigation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    navigator: SmartNavigator,
    out: Path,
    *,
    save_snapshots: bool,
) -> tuple[list[dict[str, object]], bool, float]:
    rows: list[dict[str, object]] = []
    initial_plan_saved = False
    next_snapshot_time = SNAPSHOT_INTERVAL_S
    start_time = time.perf_counter()
    reached = False

    for step in range(MAX_STEPS):
        status = navigator.step(data)
        mujoco.mj_step(model, data)
        robot_pos = navigator.get_robot_position(data)

        if navigator.path and not initial_plan_saved:
            save_path_visualization(
                navigator.path,
                navigator.grid.get_obstacles_near(robot_pos, radius=10.0),
                out / "initial_plan.png",
            )
            initial_plan_saved = True

        if save_snapshots and data.time >= next_snapshot_time:
            snapshot_name = out / f"map_snapshot_{int(next_snapshot_time):04d}.png"
            save_map_snapshot(navigator, robot_pos, navigator.path, GOAL, navigator.trajectory, snapshot_name)
            next_snapshot_time += SNAPSHOT_INTERVAL_S

        if step % SAMPLE_EVERY == 0 or status.mode == "goal_reached" or step == MAX_STEPS - 1:
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "distance_to_goal": round(float(status.distance), 4),
                    "base_x": round(float(status.base_position[0]), 4),
                    "base_y": round(float(status.base_position[1]), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "known_cells": int(np.count_nonzero(navigator.grid.grid)),
                    "obstacles": int(np.count_nonzero(navigator.grid.grid == navigator.grid.OBSTACLE)),
                    "path_points": len(navigator.path),
                }
            )

        if status.mode == "goal_reached":
            reached = True
            if not save_snapshots or data.time >= SNAPSHOT_FINAL_TIME_S:
                break

    elapsed = time.perf_counter() - start_time
    if not initial_plan_saved:
        save_path_visualization(
            navigator.path,
            navigator.grid.get_obstacles_near(navigator.get_robot_position(data), 10.0),
            out / "initial_plan.png",
        )
    return rows, reached, elapsed


def write_report(
    out: Path,
    rows: list[dict[str, object]],
    reached: bool,
    elapsed: float,
    navigator: SmartNavigator,
) -> None:
    final = rows[-1]
    report = [
        "Lab12 autonomous navigation report",
        f"goal: ({GOAL[0]:.2f}, {GOAL[1]:.2f})",
        f"reached: {reached}",
        f"final_distance: {final['distance_to_goal']}",
        f"simulation_time: {final['time']} s",
        f"wall_clock_time: {elapsed:.2f} s",
        f"path_points: {len(navigator.path)}",
        f"trajectory_samples: {len(navigator.trajectory)}",
        f"known_cells: {int(np.count_nonzero(navigator.grid.grid))}",
        f"obstacle_cells: {int(np.count_nonzero(navigator.grid.grid == navigator.grid.OBSTACLE))}",
        "artifacts: trajectory.csv, initial_plan.png, final_map.png, report.txt",
    ]
    (out / "report.txt").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out = setup_artifacts()

    if not args.no_viz and not validation_mode():
        model, data = load_lab12_model()
        viewer_controller = SmartNavigator(model, data, enable_visualization=True)
        viewer_controller.set_goal(GOAL)
        viewer_controller.reset(data)
        try:
            open_lab_viewer(
                model,
                data,
                viewer_controller,
                "laba 12: A*, DWA и карта в реальном времени",
                duration_s=60.0,
            )
        finally:
            viewer_controller.cleanup()
        return

    model, data = load_lab12_model()
    navigator = SmartNavigator(model, data, enable_visualization=False)
    navigator.set_goal(GOAL)
    navigator.reset(data)

    try:
        rows, reached, elapsed = run_navigation(model, data, navigator, out, save_snapshots=args.save_snapshots)
    finally:
        navigator.cleanup()

    save_trajectory(rows, out / "trajectory.csv")
    save_occupancy_grid(navigator.grid.grid, out / "final_map.png")
    save_map_snapshot(
        navigator,
        navigator.get_robot_position(data),
        navigator.path,
        GOAL,
        navigator.trajectory,
        out / "final_snapshot.png",
    )
    write_report(out, rows, reached, elapsed, navigator)

    print(f"Reached goal: {reached}")
    print(f"Final distance: {rows[-1]['distance_to_goal']} m")
    print(f"Simulation time: {rows[-1]['time']} s")
    print(f"Wall-clock time: {elapsed:.2f} s")
    print(f"Saved: {out / 'trajectory.csv'}")
    print(f"Saved: {out / 'initial_plan.png'}")
    print(f"Saved: {out / 'final_map.png'}")
    print(f"Saved: {out / 'report.txt'}")


if __name__ == "__main__":
    main()
