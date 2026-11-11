import heapq
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _Node:
    x: int
    y: int
    cost: float
    parent: tuple[int, int] | None


class AStarPlanner:
    def __init__(
        self,
        grid_resolution: float = 0.2,
        grid_size: tuple[int, int] = (100, 100),
        origin: tuple[float, float] | None = None,
        robot_radius: float = 0.3,
    ) -> None:
        self.grid_resolution = float(grid_resolution)
        self.grid_size = grid_size
        self.origin = origin if origin is not None else self._centered_origin(grid_size)
        self.robot_radius = float(robot_radius)
        self._motions = (
            (1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )

    def plan(self, start_pos: np.ndarray, goal_pos: np.ndarray, occupancy_grid: np.ndarray) -> list[np.ndarray]:
        grid = np.asarray(occupancy_grid)
        if grid.ndim != 2:
            raise ValueError("occupancy_grid must be a 2D array.")

        self.grid_size = (int(grid.shape[0]), int(grid.shape[1]))
        start = self.world_to_grid(float(start_pos[0]), float(start_pos[1]))
        goal = self.world_to_grid(float(goal_pos[0]), float(goal_pos[1]))
        if not self._is_free(start, grid) or not self._is_free(goal, grid):
            return []

        start_node = _Node(start[0], start[1], 0.0, None)
        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        open_nodes: dict[tuple[int, int], _Node] = {start: start_node}
        closed_nodes: dict[tuple[int, int], _Node] = {}
        heapq.heappush(open_heap, (self._heuristic(start, goal), 0, start))
        counter = 1

        while open_heap:
            _, _, current_index = heapq.heappop(open_heap)
            current = open_nodes.get(current_index)
            if current is None:
                continue

            if current_index == goal:
                return self._reconstruct_path(current, closed_nodes)

            del open_nodes[current_index]
            closed_nodes[current_index] = current

            for dx, dy, step_cost in self._motions:
                neighbor_index = (current.x + dx, current.y + dy)
                if neighbor_index in closed_nodes or not self._is_free(neighbor_index, grid):
                    continue

                new_cost = current.cost + step_cost
                existing = open_nodes.get(neighbor_index)
                if existing is not None and existing.cost <= new_cost:
                    continue

                neighbor = _Node(neighbor_index[0], neighbor_index[1], new_cost, current_index)
                open_nodes[neighbor_index] = neighbor
                priority = new_cost + self._heuristic(neighbor_index, goal)
                heapq.heappush(open_heap, (priority, counter, neighbor_index))
                counter += 1

        return []

    def world_to_grid(self, world_x: float, world_y: float) -> tuple[int, int]:
        grid_x = round((world_x - self.origin[0]) / self.grid_resolution)
        grid_y = round((world_y - self.origin[1]) / self.grid_resolution)
        return int(grid_x), int(grid_y)

    def grid_to_world(self, grid_x: int, grid_y: int) -> np.ndarray:
        world_x = grid_x * self.grid_resolution + self.origin[0]
        world_y = grid_y * self.grid_resolution + self.origin[1]
        return np.array([world_x, world_y], dtype=float)

    def _centered_origin(self, grid_size: tuple[int, int]) -> tuple[float, float]:
        return (
            -grid_size[0] * self.grid_resolution / 2.0,
            -grid_size[1] * self.grid_resolution / 2.0,
        )

    def _is_free(self, index: tuple[int, int], grid: np.ndarray) -> bool:
        x, y = index
        if x < 0 or y < 0 or x >= grid.shape[0] or y >= grid.shape[1]:
            return False
        radius_cells = math.ceil(self.robot_radius / self.grid_resolution)
        for gx in range(x - radius_cells, x + radius_cells + 1):
            for gy in range(y - radius_cells, y + radius_cells + 1):
                if gx < 0 or gy < 0 or gx >= grid.shape[0] or gy >= grid.shape[1]:
                    return False
                if grid[gx, gy] == 2:
                    dx = (gx - x) * self.grid_resolution
                    dy = (gy - y) * self.grid_resolution
                    if dx*dx + dy*dy <= self.robot_radius * self.robot_radius:
                        return False
        return True

    @staticmethod
    def _heuristic(first: tuple[int, int], second: tuple[int, int]) -> float:
        return math.hypot(first[0] - second[0], first[1] - second[1])

    def _reconstruct_path(self, goal_node: _Node, closed_nodes: dict[tuple[int, int], _Node]) -> list[np.ndarray]:
        path = [self.grid_to_world(goal_node.x, goal_node.y)]
        parent = goal_node.parent
        while parent is not None:
            node = closed_nodes[parent]
            path.append(self.grid_to_world(node.x, node.y))
            parent = node.parent
        path.reverse()
        return path
