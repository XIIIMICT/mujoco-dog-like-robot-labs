from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .occupancy_grid import OccupancyGrid


@dataclass(frozen=True)
class _MapUpdate:
    grid: np.ndarray
    robot_pos: np.ndarray
    robot_yaw: float
    path: np.ndarray
    goal: np.ndarray | None
    trajectory: np.ndarray
    current_time: float
    stats: dict[str, float | int | np.ndarray]


class MapVisualizer:
    def __init__(
        self,
        grid_size: tuple[int, int] = (100, 100),
        resolution: float = 0.2,
        origin: tuple[float, float] = (-10.0, -10.0),
        fps: float = 20.0,
        use_gui: bool = True,
    ) -> None:
        self.grid_size = grid_size
        self.resolution = float(resolution)
        self.origin = origin
        self.fps = float(fps)
        self.use_gui = use_gui
        self._updates: queue.Queue[_MapUpdate] = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._last_gui_draw = 0.0
        self._figure = None
        self._axis = None
        self._elements: dict[str, object] = {}
        self._disabled_reason = ""
        self._last_grid: np.ndarray | None = None

    def start(self) -> None:
        self._started = True
        self._last_gui_draw = 0.0
        if self.use_gui:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_visualization, name="MapVisualizer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self.use_gui and self._figure is not None:
            plt.close(self._figure)
            self._figure = None
            self._axis = None
            self._elements.clear()
            self._last_grid = None
        self._started = False
        self._last_gui_draw = 0.0

    def update(
        self,
        occupancy_grid: OccupancyGrid,
        robot_pos: np.ndarray,
        robot_yaw: float,
        path: list[np.ndarray] | np.ndarray,
        goal: np.ndarray | None,
        trajectory: list[np.ndarray] | np.ndarray,
        current_time: float,
    ) -> None:
        if self._disabled_reason:
            return

        if self.use_gui:
            if not self._started:
                self.start()
            now = time.perf_counter()
            min_interval = 1.0 / self.fps if self.fps > 0.0 else 0.0
            if now - self._last_gui_draw < min_interval:
                return
            if self._figure is None:
                self._init_map_elements()
            self._update_visualization(
                _MapUpdate(
                    grid=occupancy_grid.grid.copy(),
                    robot_pos=np.asarray(robot_pos, dtype=float).copy(),
                    robot_yaw=float(robot_yaw),
                    path=self._as_points(path),
                    goal=None if goal is None else np.asarray(goal, dtype=float).copy(),
                    trajectory=self._as_points(trajectory),
                    current_time=float(current_time),
                    stats=self._grid_stats(occupancy_grid.grid),
                ),
            )
            self._last_gui_draw = now
            plt.pause(0.001)
            return

        update = _MapUpdate(
            grid=occupancy_grid.grid.copy(),
            robot_pos=np.asarray(robot_pos, dtype=float).copy(),
            robot_yaw=float(robot_yaw),
            path=self._as_points(path),
            goal=None if goal is None else np.asarray(goal, dtype=float).copy(),
            trajectory=self._as_points(trajectory),
            current_time=float(current_time),
            stats=self._grid_stats(occupancy_grid.grid),
        )
        self._replace_pending_update(update)

    def _run_visualization(self) -> None:
        try:
            if not self.use_gui:
                mpl.use("Agg", force=True)
            plt.ion()
            self._init_map_elements()
        except Exception as exc:  # noqa: BLE001
            self._disabled_reason = str(exc)
            print(f"MapVisualizer disabled: {exc}")
            return

        frame_delay = 1.0 / self.fps if self.fps > 0.0 else 0.05
        while not self._stop_event.is_set():
            try:
                update = self._updates.get(timeout=frame_delay)
            except queue.Empty:
                plt.pause(0.001)
                continue
            self._update_visualization(update)
            plt.pause(0.001)
        plt.close(self._figure)

    def _init_map_elements(self) -> None:
        self._figure, self._axis = plt.subplots(figsize=(3.2, 2.4), num="Lab12 Occupancy Map")
        self._position_window()
        extent = self._map_extent()
        empty_grid = np.zeros(self.grid_size, dtype=np.int8)
        map_image = self._axis.imshow(
            empty_grid.T,
            origin="lower",
            extent=extent,
            vmin=0,
            vmax=2,
            cmap="Greys",
            interpolation="nearest",
        )
        (robot_marker,) = self._axis.plot([], [], "bo", markersize=7, label="Robot")
        (robot_direction,) = self._axis.plot([], [], "b-", linewidth=2)
        (trajectory_line,) = self._axis.plot([], [], "c-", linewidth=1.5, label="Trajectory")
        (path_line,) = self._axis.plot([], [], "r--", linewidth=1.2, label="A* path")
        (goal_marker,) = self._axis.plot([], [], "gx", markersize=9, label="Goal")
        stats_text = self._axis.text(0.02, 0.98, "", transform=self._axis.transAxes, va="top")

        self._axis.set_title("Lab12 Real-time Occupancy Map")
        self._axis.set_xlabel("x [m]")
        self._axis.set_ylabel("y [m]")
        self._axis.set_aspect("equal", adjustable="box")
        self._axis.grid(visible=True, alpha=0.25)
        self._axis.legend(loc="lower right")
        self._elements = {
            "map_image": map_image,
            "robot_marker": robot_marker,
            "robot_direction": robot_direction,
            "trajectory_line": trajectory_line,
            "path_line": path_line,
            "goal_marker": goal_marker,
            "stats_text": stats_text,
        }
        self._figure.tight_layout()
        self._figure.canvas.draw_idle()

    def _position_window(self) -> None:
        manager = getattr(self._figure.canvas, "manager", None)
        window = getattr(manager, "window", None)
        if window is None:
            return
        with contextlib.suppress(Exception):
            window.wm_geometry("650x780+1240+180")

    def _update_visualization(self, data: _MapUpdate) -> None:
        if self._last_grid is None or not np.array_equal(self._last_grid, data.grid):
            self._elements["map_image"].set_data(data.grid.T)
            self._last_grid = data.grid.copy()
        self._elements["robot_marker"].set_data([data.robot_pos[0]], [data.robot_pos[1]])

        direction_length = 0.45
        direction_x = [data.robot_pos[0], data.robot_pos[0] + direction_length * np.cos(data.robot_yaw)]
        direction_y = [data.robot_pos[1], data.robot_pos[1] + direction_length * np.sin(data.robot_yaw)]
        self._elements["robot_direction"].set_data(direction_x, direction_y)

        self._set_line_points("trajectory_line", data.trajectory)
        self._set_line_points("path_line", data.path)
        if data.goal is None:
            self._elements["goal_marker"].set_data([], [])
        else:
            self._elements["goal_marker"].set_data([data.goal[0]], [data.goal[1]])

        self._elements["stats_text"].set_text(
            "time: {:.1f}s\nexplored: {:.1f}%\nfree: {}\nobstacles: {}".format(
                data.current_time,
                float(data.stats["explored_ratio"]) * 100.0,
                data.stats["free_count"],
                data.stats["obstacle_count"],
            ),
        )
        self._figure.canvas.draw_idle()

    def _replace_pending_update(self, update: _MapUpdate) -> None:
        try:
            self._updates.put_nowait(update)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._updates.get_nowait()
            self._updates.put_nowait(update)

    def _map_extent(self) -> tuple[float, float, float, float]:
        min_x = self.origin[0]
        min_y = self.origin[1]
        max_x = min_x + self.grid_size[0] * self.resolution
        max_y = min_y + self.grid_size[1] * self.resolution
        return min_x, max_x, min_y, max_y

    def _set_line_points(self, element_name: str, points: np.ndarray) -> None:
        if points.size == 0:
            self._elements[element_name].set_data([], [])
            return
        self._elements[element_name].set_data(points[:, 0], points[:, 1])

    @staticmethod
    def _grid_stats(grid: np.ndarray) -> dict[str, float | int]:
        total_count = int(grid.size)
        known_count = int(np.count_nonzero(grid != 0))
        return {
            "free_count": int(np.count_nonzero(grid == 1)),
            "obstacle_count": int(np.count_nonzero(grid == 2)),
            "explored_ratio": known_count / total_count if total_count else 0.0,
        }

    @staticmethod
    def _as_points(points: list[np.ndarray] | np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.empty((0, 2), dtype=float)
        point_array = np.asarray(points, dtype=float)
        return point_array.reshape(-1, point_array.shape[-1])[:, :2]

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    @property
    def disabled_reason(self) -> str:
        return self._disabled_reason
