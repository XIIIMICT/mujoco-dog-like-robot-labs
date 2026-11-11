import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class _TrajectoryScore:
    cost: float
    velocity: np.ndarray
    trajectory: np.ndarray


class DWAController:
    def __init__(
        self,
        max_speed: float = 1.0,
        max_yaw_rate: float = 1.0,
        max_accel: float = 0.5,
        max_delta_yaw_rate: float = 0.5,
        velocity_resolution: float = 0.05,
        yaw_rate_resolution: float = 0.1,
        dt: float = 0.1,
        predict_time: float = 2.0,
        to_goal_cost_gain: float = 1.0,
        speed_cost_gain: float = 1.0,
        obstacle_cost_gain: float = 2.0,
        robot_radius: float = 0.3,
    ) -> None:
        self.max_speed = float(max_speed)
        self.min_speed = 0.0
        self.max_yaw_rate = float(max_yaw_rate)
        self.max_accel = float(max_accel)
        self.max_delta_yaw_rate = float(max_delta_yaw_rate)
        self.velocity_resolution = float(velocity_resolution)
        self.yaw_rate_resolution = float(yaw_rate_resolution)
        self.dt = float(dt)
        self.predict_time = float(predict_time)
        self.to_goal_cost_gain = float(to_goal_cost_gain)
        self.speed_cost_gain = float(speed_cost_gain)
        self.obstacle_cost_gain = float(obstacle_cost_gain)
        self.robot_radius = float(robot_radius)

    def compute_velocity(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: list[list[float]] | np.ndarray,
    ) -> np.ndarray:
        dynamic_window = self.calculate_dynamic_window(state)
        obstacle_array = self._as_obstacle_array(obstacles)
        best = self._search_best_trajectory(state, goal, obstacle_array, dynamic_window)
        return best.velocity

    def calculate_dynamic_window(self, current_velocity: np.ndarray) -> tuple[float, float, float, float]:
        current_speed = float(current_velocity[3])
        current_yaw_rate = float(current_velocity[4])
        speed_min = max(self.min_speed, current_speed - self.max_accel * self.dt)
        speed_max = min(self.max_speed, current_speed + self.max_accel * self.dt)
        yaw_rate_min = max(-self.max_yaw_rate, current_yaw_rate - self.max_delta_yaw_rate * self.dt)
        yaw_rate_max = min(self.max_yaw_rate, current_yaw_rate + self.max_delta_yaw_rate * self.dt)
        return speed_min, speed_max, yaw_rate_min, yaw_rate_max

    def predict_trajectory(self, state: np.ndarray, v: float, omega: float) -> np.ndarray:
        predicted = np.array(state, dtype=float)
        trajectory = [predicted.copy()]
        time = 0.0
        while time < self.predict_time:
            predicted = self._motion(predicted, v, omega)
            trajectory.append(predicted.copy())
            time += self.dt
        return np.array(trajectory, dtype=float)

    def calc_obstacle_cost(self, trajectory: np.ndarray, obstacles: list[list[float]] | np.ndarray) -> float:
        obstacle_array = self._as_obstacle_array(obstacles)
        if obstacle_array.size == 0:
            return 0.0

        deltas = trajectory[:, None, 0:2] - obstacle_array[None, :, 0:2]
        distances = np.linalg.norm(deltas, axis=2)
        min_distance = float(np.min(distances))
        if min_distance <= self.robot_radius:
            return float("inf")
        return 1.0 / min_distance

    def _search_best_trajectory(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        dynamic_window: tuple[float, float, float, float],
    ) -> _TrajectoryScore:
        best = _TrajectoryScore(
            cost=float("inf"),
            velocity=np.zeros(3, dtype=float),
            trajectory=np.array([state], dtype=float),
        )
        for speed in self._sample_range(dynamic_window[0], dynamic_window[1], self.velocity_resolution):
            for yaw_rate in self._sample_range(dynamic_window[2], dynamic_window[3], self.yaw_rate_resolution):
                trajectory = self.predict_trajectory(state, speed, yaw_rate)
                obstacle_cost = self.calc_obstacle_cost(trajectory, obstacles)
                if math.isinf(obstacle_cost):
                    continue

                to_goal_cost = self.to_goal_cost_gain * self._calc_to_goal_cost(trajectory, goal)
                speed_cost = self.speed_cost_gain * (self.max_speed - speed)
                final_cost = to_goal_cost + speed_cost + self.obstacle_cost_gain * obstacle_cost
                if final_cost < best.cost:
                    velocity = self._local_velocity_command(speed, yaw_rate, state, goal)
                    best = _TrajectoryScore(final_cost, velocity, trajectory)

        if math.isinf(best.cost):
            best = _TrajectoryScore(
                cost=0.0,
                velocity=np.array([0.0, 0.0, 0.0], dtype=float),
                trajectory=np.array([state], dtype=float),
            )
        return best

    def _motion(self, state: np.ndarray, speed: float, yaw_rate: float) -> np.ndarray:
        next_state = np.array(state, dtype=float)
        next_state[2] += yaw_rate * self.dt
        next_state[0] += speed * math.cos(next_state[2]) * self.dt
        next_state[1] += speed * math.sin(next_state[2]) * self.dt
        next_state[3] = speed
        next_state[4] = yaw_rate
        return next_state

    def _local_velocity_command(self, speed: float, yaw_rate: float, state: np.ndarray, goal: np.ndarray) -> np.ndarray:
        desired_yaw = math.atan2(float(goal[1] - state[1]), float(goal[0] - state[0]))
        yaw_error = self._wrap_angle(desired_yaw - float(state[2]))
        vx = speed * math.cos(yaw_error)
        vy = speed * math.sin(yaw_error)
        return np.array([vx, vy, yaw_rate], dtype=float)

    @staticmethod
    def _calc_to_goal_cost(trajectory: np.ndarray, goal: np.ndarray) -> float:
        dx = float(goal[0] - trajectory[-1, 0])
        dy = float(goal[1] - trajectory[-1, 1])
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - float(trajectory[-1, 2])
        return abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    @staticmethod
    def _sample_range(start: float, stop: float, step: float) -> np.ndarray:
        if math.isclose(start, stop):
            return np.array([start], dtype=float)
        samples = np.arange(start, stop + step * 0.5, step, dtype=float)
        return np.clip(samples, min(start, stop), max(start, stop))

    @staticmethod
    def _as_obstacle_array(obstacles: list[list[float]] | np.ndarray) -> np.ndarray:
        obstacle_array = np.asarray(obstacles, dtype=float)
        if obstacle_array.size == 0:
            return np.empty((0, 2), dtype=float)
        return obstacle_array.reshape(-1, 2)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi
