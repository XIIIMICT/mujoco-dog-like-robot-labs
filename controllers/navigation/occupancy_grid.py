import math

import numpy as np


class OccupancyGrid:
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2

    def __init__(
        self,
        size: tuple[int, int] = (100, 100),
        resolution: float = 0.2,
        origin: tuple[float, float] | None = None,
    ) -> None:
        self.size = size
        self.resolution = float(resolution)
        self.origin = origin if origin is not None else self._centered_origin(size)
        self.grid = np.zeros(size, dtype=np.int8)
        self.confidence = np.zeros(size, dtype=np.float32)
        self.discovery_time = np.full(size, -1.0, dtype=np.float32)
        self.update_count = np.zeros(size, dtype=np.int32)

    def update_with_ray(
        self,
        robot_pos: np.ndarray,
        ray_angle: float,
        ray_distance: float,
        max_range: float,
        current_time: float,
    ) -> None:
        start_x, start_y = self.world_to_grid(float(robot_pos[0]), float(robot_pos[1]))
        clipped_distance = min(float(ray_distance), float(max_range))
        end_world_x = float(robot_pos[0]) + clipped_distance * math.cos(ray_angle)
        end_world_y = float(robot_pos[1]) + clipped_distance * math.sin(ray_angle)
        end_x, end_y = self.world_to_grid(end_world_x, end_world_y)
        ray_cells = self._bresenham_line(start_x, start_y, end_x, end_y)
        if not ray_cells:
            return

        hit_obstacle = ray_distance < max_range
        free_cells = ray_cells[:-1] if hit_obstacle else ray_cells
        for x, y in free_cells:
            self.mark_free(x, y, current_time)

        if hit_obstacle:
            self.mark_obstacle(ray_cells[-1][0], ray_cells[-1][1], current_time)

    def mark_free(self, x: int, y: int, timestamp: float) -> None:
        if not self.is_valid_index(x, y):
            return
        if self.grid[x, y] != self.OBSTACLE:
            self.grid[x, y] = self.FREE
        self._record_update(x, y, timestamp, confidence_delta=0.05)

    def mark_obstacle(self, x: int, y: int, timestamp: float) -> None:
        if not self.is_valid_index(x, y):
            return
        self.grid[x, y] = self.OBSTACLE
        self._record_update(x, y, timestamp, confidence_delta=0.25)

    def get_visualization_data(self) -> dict[str, float | int | np.ndarray]:
        known_count = int(np.count_nonzero(self.grid != self.UNKNOWN))
        total_count = int(self.grid.size)
        return {
            "grid": self.grid.copy(),
            "confidence": self.confidence.copy(),
            "discovery_time": self.discovery_time.copy(),
            "free_count": int(np.count_nonzero(self.grid == self.FREE)),
            "obstacle_count": int(np.count_nonzero(self.grid == self.OBSTACLE)),
            "unknown_count": int(np.count_nonzero(self.grid == self.UNKNOWN)),
            "explored_ratio": known_count / total_count if total_count else 0.0,
        }

    def _bresenham_line(self, x0: int, y0: int, x1: int, y1: int) -> list[tuple[int, int]]:
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0

        while True:
            if self.is_valid_index(x, y):
                cells.append((x, y))
            if x == x1 and y == y1:
                break
            err2 = 2 * err
            if err2 > -dy:
                err -= dy
                x += sx
            if err2 < dx:
                err += dx
                y += sy
        return cells

    def world_to_grid(self, wx: float, wy: float) -> tuple[int, int]:
        gx = round((wx - self.origin[0]) / self.resolution)
        gy = round((wy - self.origin[1]) / self.resolution)
        return int(gx), int(gy)

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        wx = gx * self.resolution + self.origin[0]
        wy = gy * self.resolution + self.origin[1]
        return wx, wy

    def is_valid_index(self, x: int, y: int) -> bool:
        return 0 <= x < self.size[0] and 0 <= y < self.size[1]

    def is_collision(self, x: float, y: float, robot_radius: float) -> bool:
        center_x, center_y = self.world_to_grid(x, y)
        radius_cells = math.ceil(robot_radius / self.resolution)
        for gx in range(center_x - radius_cells, center_x + radius_cells + 1):
            for gy in range(center_y - radius_cells, center_y + radius_cells + 1):
                if not self.is_valid_index(gx, gy):
                    return True
                if self.grid[gx, gy] == self.OBSTACLE:
                    wx, wy = self.grid_to_world(gx, gy)
                    if math.hypot(wx - x, wy - y) <= robot_radius:
                        return True
        return False

    def get_obstacles_near(self, pos: np.ndarray, radius: float) -> list[tuple[float, float]]:
        center_x, center_y = self.world_to_grid(float(pos[0]), float(pos[1]))
        radius_cells = math.ceil(radius / self.resolution)
        obstacles = []
        for gx in range(center_x - radius_cells, center_x + radius_cells + 1):
            for gy in range(center_y - radius_cells, center_y + radius_cells + 1):
                if self.is_valid_index(gx, gy) and self.grid[gx, gy] == self.OBSTACLE:
                    wx, wy = self.grid_to_world(gx, gy)
                    if math.hypot(wx - float(pos[0]), wy - float(pos[1])) <= radius:
                        obstacles.append((wx, wy))
        return obstacles

    def _record_update(self, x: int, y: int, timestamp: float, confidence_delta: float) -> None:
        if self.discovery_time[x, y] < 0.0:
            self.discovery_time[x, y] = float(timestamp)
        self.update_count[x, y] += 1
        self.confidence[x, y] = min(1.0, float(self.confidence[x, y]) + confidence_delta)

    def _centered_origin(self, size: tuple[int, int]) -> tuple[float, float]:
        return (
            -size[0] * self.resolution / 2.0,
            -size[1] * self.resolution / 2.0,
        )
