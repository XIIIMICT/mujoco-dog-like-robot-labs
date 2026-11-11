import math
from typing import TYPE_CHECKING, cast

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

from controllers.common import (
    DEFAULT_POLICY,
    DEFAULT_POLICY_CONFIG,
    ControllerStatus,
    joint_addresses_for_actuators,
    object_id,
    quat_conjugate_rotate,
    wrap_angle,
    yaw_from_quat,
)

from .astar_planner import AStarPlanner
from .dwa_controller import DWAController
from .map_visualizer import MapVisualizer
from .occupancy_grid import OccupancyGrid

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class SmartNavigator:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        enable_visualization: bool = True,
        policy_path: Path = DEFAULT_POLICY,
        config_path: Path = DEFAULT_POLICY_CONFIG,
        grid_size: tuple[int, int] = (100, 100),
        grid_resolution: float = 0.2,
        grid_origin: tuple[float, float] = (-10.0, -10.0),
        goal_tolerance: float = 0.3,
        robot_radius: float = 0.3,
    ) -> None:
        self._model = model
        self._base_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._goal = np.array([6.0, 0.0], dtype=float)
        self._goal_tolerance = float(goal_tolerance)
        self._robot_radius = float(robot_radius)
        self._navigation_margin = 0.015
        self._navigation_radius = self._robot_radius + self._navigation_margin
        self._waypoint_radius = 0.45
        self._cruise_speed = 0.6
        self._turn_in_place_angle = 0.55
        self._narrow_passage_distance = 0.68
        self._robot_half_width = 0.17

        self.grid = OccupancyGrid(size=grid_size, resolution=grid_resolution, origin=grid_origin)
        self.astar = AStarPlanner(
            grid_resolution=grid_resolution,
            grid_size=grid_size,
            origin=grid_origin,
            robot_radius=self._navigation_radius,
        )
        self.dwa = DWAController(max_speed=1.0, max_yaw_rate=1.0, robot_radius=robot_radius)
        self.visualizer = (
            MapVisualizer(grid_size=grid_size, resolution=grid_resolution, origin=grid_origin)
            if enable_visualization
            else None
        )

        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self._policy_dt = float(cfg["step_dt"])
        self._joint_map = np.array(cfg["joint_ids_map"], dtype=np.int32)
        self._default_joint_pos = np.array(cfg["default_joint_pos"], dtype=np.float32)
        self._stiffness = np.array(cfg["stiffness"], dtype=np.float32)
        self._damping = np.array(cfg["damping"], dtype=np.float32)
        action_cfg = cfg["actions"]["JointPositionAction"]
        self._action_scale = np.array(action_cfg["scale"], dtype=np.float32)
        self._action_offset = np.array(action_cfg["offset"], dtype=np.float32)

        self._qpos_adr, self._qvel_adr = joint_addresses_for_actuators(model, self._joint_map)
        self._ctrl_low = model.actuator_ctrlrange[:, 0]
        self._ctrl_high = model.actuator_ctrlrange[:, 1]
        self._gyro_sensor_adr = self._sensor_adr(model, ("imu_gyro", "imu_ang_vel"))
        self._session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        self._gravity_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        self._command = np.zeros(3, dtype=np.float32)
        self._command_target = np.zeros(3, dtype=np.float32)
        self._obs = np.zeros((1, 45), dtype=np.float32)
        self._last_action = np.zeros(12, dtype=np.float32)
        self._target_joint_pos = self._default_joint_pos.copy()
        self._next_policy_time = 0.0

        self.path: list[np.ndarray] = []
        self.trajectory: list[np.ndarray] = []
        self.predicted_trajectory: np.ndarray = np.empty((0, 2), dtype=float)
        self._path_target_index = 0
        self._last_replan_time = -math.inf
        self._replan_interval = 1.5
        self._grid_change_replan_delay = 0.25
        self._local_target_stride = 4
        self._visualization_interval = 0.2
        self._last_visualization_time = -math.inf
        self._visualization_trail_limit = 400
        self._emergency_stop_distance = 0.65
        self._avoid_duration = 0.85
        self._avoid_forward_speed = 0.52
        self._avoid_lateral_speed = 0.30
        self._avoid_turn_rate = 0.42
        self._avoid_stall_distance = 0.18
        self._progress_timeout = 1.4
        self._progress_distance = 0.35
        self._front_sector_angle = math.radians(20.0)
        self._mapping_sector_angle = math.radians(120.0)
        self._body_clearance_distance = 0.4
        self._scan_fov = math.radians(180.0)
        self._scan_angle_step = math.radians(12.0)
        self._front_ray_angles = self._build_scan_angles()
        self._max_scan_range = 6.0
        self._ray_geomid = np.array([-1], dtype=np.int32)
        self._ray_origin = np.zeros(3, dtype=np.float64)
        self._ray_direction = np.zeros(3, dtype=np.float64)
        self._progress_reference_pos = np.zeros(2, dtype=float)
        self._progress_reference_time = 0.0
        self._avoid_until = -math.inf
        self._avoid_side = 1.0
        self._last_grid_change_time = -math.inf
        self._last_known_cells = 0
        self._last_status = ControllerStatus("nav_init", "goal", math.inf, 0.0, (0.0, 0.0, 0.0))
        mujoco.mj_forward(self._model, data)

    def reset(self, data: mujoco.MjData) -> None:
        self._command[:] = 0.0
        self._command_target[:] = 0.0
        self._last_action[:] = 0.0
        self._target_joint_pos[:] = self._default_joint_pos
        self._next_policy_time = 0.0
        self.path = []
        self.trajectory = []
        self.predicted_trajectory = np.empty((0, 2), dtype=float)
        self._path_target_index = 0
        self._last_replan_time = -math.inf
        self._last_visualization_time = -math.inf
        self._progress_reference_pos = self.get_robot_position(data)
        self._progress_reference_time = float(data.time)
        self._avoid_until = -math.inf
        self._avoid_side = 1.0
        self._last_grid_change_time = -math.inf
        self._last_known_cells = 0
        if self.visualizer is not None:
            self.visualizer.start()
        mujoco.mj_forward(self._model, data)

    def set_goal(self, goal_pos: np.ndarray) -> None:
        self._goal = np.asarray(goal_pos, dtype=float).copy()
        self.path = []
        self._path_target_index = 0
        self._last_replan_time = -math.inf
        self._last_visualization_time = -math.inf
        self._avoid_until = -math.inf
        self._avoid_side = 1.0
        self._last_grid_change_time = -math.inf
        self._last_known_cells = 0

    def step(self, mujoco_data: mujoco.MjData) -> ControllerStatus:
        status = self._navigation_status(mujoco_data)
        if self.is_goal_reached():
            self._command_target[:] = 0.0
            self._command[:] = 0.7 * self._command
            self._apply_policy(mujoco_data, self._command)
            self._last_status = self._status_with_mode(status, "goal_reached")
            return self._last_status

        command = self._navigation_command(mujoco_data)
        self._command_target[:] = command.astype(np.float32)
        self._command[:] = 0.65 * self._command + 0.35 * self._command_target
        self._apply_policy(mujoco_data, self._command)
        self._last_status = self._navigation_status(mujoco_data)
        return self._last_status

    def update_map(self, robot_pos: np.ndarray, scan_data: Sequence[tuple[float, float]], current_time: float) -> None:
        for ray_angle, ray_distance in scan_data:
            self.grid.update_with_ray(robot_pos, ray_angle, ray_distance, self._max_scan_range, current_time)

    def check_replan_needed(self) -> bool:
        return not self.path

    def replan_path(self, robot_pos: np.ndarray) -> list[np.ndarray]:
        path = self.astar.plan(robot_pos, self._goal, self.grid.grid)
        if path:
            self.path = path
            self._path_target_index = self._closest_path_index(robot_pos)
        return self.path

    def get_current_target(self) -> np.ndarray:
        if not self.path:
            return self._goal
        return self.path[min(self._path_target_index, len(self.path) - 1)]

    def get_local_goal(self) -> np.ndarray:
        if not self.path:
            return self._goal
        index = min(self._path_target_index + self._local_target_stride, len(self.path) - 1)
        return self.path[index]

    def is_goal_reached(self) -> bool:
        if self._last_status.distance == math.inf:
            return False
        return self._last_status.distance <= self._goal_tolerance

    def cleanup(self) -> None:
        if self.visualizer is not None:
            self.visualizer.stop()

    def _navigation_command(self, data: mujoco.MjData) -> np.ndarray:
        robot_pos = self.get_robot_position(data)
        robot_yaw = self.get_robot_yaw(data)
        known_before = int(np.count_nonzero(self.grid.grid))
        scan = self._scan_obstacles(data, robot_yaw)
        self._update_occupancy_grid(robot_pos, robot_yaw, scan, float(data.time))
        self._update_grid_change_tracking(known_before, float(data.time))
        self.trajectory.append(robot_pos.copy())
        front_distance = self._front_distance(scan)
        current_time = float(data.time)
        nearby_obstacles = self._filtered_obstacles_near(robot_pos, radius=3.0)
        body_clearance = self._body_clearance(robot_pos, nearby_obstacles)
        stuck = self._is_stuck(robot_pos, current_time)

        if self.check_replan_needed() or self._should_replan(robot_pos, front_distance, body_clearance, current_time):
            self.replan_path(robot_pos)
            self._last_replan_time = current_time
        self._advance_path_target(robot_pos)

        local_goal = self.get_local_goal()
        state = np.array([robot_pos[0], robot_pos[1], robot_yaw, self._command[0], self._command[2]], dtype=float)
        blocked = front_distance <= self._emergency_stop_distance or body_clearance <= self._body_clearance_distance
        if blocked:
            self._avoid_side = self._choose_avoid_side(scan)
            self._avoid_until = max(self._avoid_until, current_time + self._avoid_duration)
        elif stuck:
            self._avoid_side = self._choose_avoid_side(scan)
            self._avoid_until = max(self._avoid_until, current_time + self._avoid_duration)

        if (
            front_distance > self._emergency_stop_distance * 1.15
            and body_clearance > self._body_clearance_distance * 1.05
        ):
            self._avoid_until = min(self._avoid_until, current_time)

        if current_time < self._avoid_until:
            command = self._avoid_command(local_goal, robot_yaw, scan, body_clearance)
        else:
            command = self._path_follow_command(robot_pos, robot_yaw, local_goal, body_clearance, front_distance)
            dwa_command = self.dwa.compute_velocity(state, local_goal, nearby_obstacles)
            if abs(float(dwa_command[2])) > 0.05:
                command[2] = 0.85 * command[2] + 0.15 * dwa_command[2]

        self._update_progress(robot_pos, current_time, command)
        speed = float(np.hypot(command[0], command[1]))
        self.predicted_trajectory = self.dwa.predict_trajectory(state, speed, float(command[2]))[:, :2]
        self._update_visualization(robot_pos, robot_yaw, current_time)
        return command

    def _front_distance(self, scan: Sequence[tuple[float, float]]) -> float:
        front_distances = [distance for angle, distance in scan if abs(angle) <= self._front_sector_angle]
        if not front_distances:
            return self._max_scan_range
        return min(front_distances)

    def _build_scan_angles(self) -> tuple[float, ...]:
        half_fov = self._scan_fov / 2.0
        steps = max(1, round(self._scan_fov / self._scan_angle_step))
        return tuple(np.linspace(-half_fov, half_fov, steps + 1, dtype=float))

    def _choose_avoid_side(self, scan: Sequence[tuple[float, float]]) -> float:
        left_clearance = self._sector_distance(scan, math.radians(25.0), math.radians(110.0))
        right_clearance = self._sector_distance(scan, math.radians(-110.0), math.radians(-25.0))
        return 1.0 if left_clearance >= right_clearance else -1.0

    @staticmethod
    def _sector_distance(scan: Sequence[tuple[float, float]], min_angle: float, max_angle: float) -> float:
        distances = [distance for angle, distance in scan if min_angle <= angle <= max_angle]
        if not distances:
            return math.inf
        return min(distances)

    def _filtered_obstacles_near(self, robot_pos: np.ndarray, radius: float) -> list[tuple[float, float]]:
        min_distance = self._navigation_radius
        obstacles = [
            obstacle
            for obstacle in self.grid.get_obstacles_near(robot_pos, radius)
            if math.hypot(obstacle[0] - float(robot_pos[0]), obstacle[1] - float(robot_pos[1])) > min_distance
        ]
        obstacles.sort(
            key=lambda obstacle: math.hypot(
                obstacle[0] - float(robot_pos[0]),
                obstacle[1] - float(robot_pos[1]),
            ),
        )
        return obstacles[:12]

    def _body_clearance(self, robot_pos: np.ndarray, nearby_obstacles: list[tuple[float, float]]) -> float:
        if not nearby_obstacles:
            return math.inf
        return min(
            math.hypot(obstacle[0] - float(robot_pos[0]), obstacle[1] - float(robot_pos[1]))
            for obstacle in nearby_obstacles
        )

    def _should_replan(
        self,
        robot_pos: np.ndarray,
        front_distance: float,
        body_clearance: float,
        current_time: float,
    ) -> bool:
        if current_time - self._last_replan_time < self._replan_interval:
            return False
        if 0.0 <= current_time - self._last_grid_change_time <= self._grid_change_replan_delay:
            return True
        if front_distance <= self._emergency_stop_distance:
            return True
        if body_clearance <= self._body_clearance_distance:
            return True
        return self._path_blocked(robot_pos)

    def _update_grid_change_tracking(self, known_before: int, current_time: float) -> None:
        known_after = int(np.count_nonzero(self.grid.grid))
        self._last_known_cells = known_after
        if known_after > known_before:
            self._last_grid_change_time = current_time

    def _path_blocked(self, robot_pos: np.ndarray) -> bool:
        if not self.path:
            return True
        local_goal = self.get_local_goal()
        distance = float(np.linalg.norm(local_goal - robot_pos))
        samples = max(2, int(distance / max(self.grid.resolution * 0.5, 1e-6)))
        for fraction in np.linspace(0.0, 1.0, samples):
            point = robot_pos * (1.0 - fraction) + local_goal * fraction
            if self.grid.is_collision(float(point[0]), float(point[1]), self._navigation_radius):
                return True
        return False

    def _closest_path_index(self, robot_pos: np.ndarray) -> int:
        if not self.path:
            return 0
        distances = [float(np.linalg.norm(point - robot_pos)) for point in self.path]
        return int(np.argmin(distances))

    def _path_follow_command(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        local_goal: np.ndarray,
        body_clearance: float,
        front_distance: float,
    ) -> np.ndarray:
        error = local_goal - robot_pos
        distance = float(np.linalg.norm(error))
        if distance <= self._goal_tolerance:
            return np.zeros(3, dtype=float)

        desired_yaw = math.atan2(float(error[1]), float(error[0]))
        yaw_error = wrap_angle(desired_yaw - robot_yaw)
        speed = self._cruise_speed
        narrow_ratio = float(
            np.clip(
                (self._narrow_passage_distance - body_clearance) / max(self._narrow_passage_distance, 1e-6),
                0.0,
                1.0,
            ),
        )
        yaw_rate = float(np.clip((1.1 - 0.20 * narrow_ratio) * yaw_error, -0.55, 0.55))
        turn_in_place_angle = self._turn_in_place_angle + 0.25 * narrow_ratio
        if abs(yaw_error) > turn_in_place_angle:
            vx = 0.18 if front_distance > self._emergency_stop_distance * 1.2 else 0.0
            vy = 0.0
        else:
            vx = speed * math.cos(yaw_error)
            lateral_scale = 0.08 + 0.20 * narrow_ratio
            lateral_limit = 0.03 + 0.07 * narrow_ratio
            vy = float(np.clip(lateral_scale * speed * math.sin(yaw_error), -lateral_limit, lateral_limit))
        return np.array([vx, vy, yaw_rate], dtype=float)

    def _avoid_command(
        self,
        local_goal: np.ndarray,
        robot_yaw: float,
        scan: Sequence[tuple[float, float]],
        body_clearance: float,
    ) -> np.ndarray:
        desired_yaw = math.atan2(float(local_goal[1]), float(local_goal[0]))
        yaw_error = wrap_angle(desired_yaw - robot_yaw)
        front_distance = self._front_distance(scan)
        effective_clearance = body_clearance - self._robot_half_width
        if front_distance <= self._avoid_stall_distance:
            self._avoid_side = self._choose_avoid_side(scan)
            vy = self._avoid_side * self._avoid_lateral_speed * 0.5
            return np.array([0.0, vy, self._avoid_turn_rate * self._avoid_side], dtype=float)
        narrow_ratio = float(
            np.clip(
                (self._narrow_passage_distance - body_clearance) / max(self._narrow_passage_distance, 1e-6),
                0.0,
                1.0,
            ),
        )
        if effective_clearance < 0.15:
            vx = 0.0
        else:
            clearance_factor = min(1.0, max(0.0, effective_clearance) / (self._body_clearance_distance * 2.0))
            vx = self._avoid_forward_speed * max(0.20, min(front_distance / self._emergency_stop_distance, 1.0)) * clearance_factor
        vy = self._avoid_side * self._avoid_lateral_speed * (0.18 + 0.30 * narrow_ratio)
        yaw_rate = float(
            np.clip(
                (0.45 - 0.10 * narrow_ratio) * self._avoid_side + 0.22 * yaw_error,
                -self._avoid_turn_rate,
                self._avoid_turn_rate,
            ),
        )
        return np.array([vx, vy, yaw_rate], dtype=float)

    def _is_stuck(self, robot_pos: np.ndarray, current_time: float) -> bool:
        if current_time - self._progress_reference_time < self._progress_timeout:
            return False
        progress = float(np.linalg.norm(robot_pos - self._progress_reference_pos))
        return progress < self._progress_distance

    def _update_progress(self, robot_pos: np.ndarray, current_time: float, command: np.ndarray) -> None:
        if current_time < self._avoid_until:
            return
        command_speed = float(np.linalg.norm(command[:2]))
        if command_speed < 0.05:
            self._progress_reference_pos = robot_pos.copy()
            self._progress_reference_time = current_time
            return
        if float(np.linalg.norm(robot_pos - self._progress_reference_pos)) >= self._progress_distance:
            self._progress_reference_pos = robot_pos.copy()
            self._progress_reference_time = current_time

    def _fallback_local_command(self, robot_pos: np.ndarray, robot_yaw: float, local_goal: np.ndarray) -> np.ndarray:
        return self._path_follow_command(robot_pos, robot_yaw, local_goal, math.inf, self._max_scan_range)

    def _update_occupancy_grid(
        self,
        robot_pos: np.ndarray,
        robot_yaw: float,
        scan: Sequence[tuple[float, float]],
        current_time: float,
    ) -> None:
        for local_angle, ray_distance in scan:
            if abs(local_angle) > self._mapping_sector_angle:
                continue
            world_angle = robot_yaw + local_angle
            self.grid.update_with_ray(robot_pos, world_angle, ray_distance, self._max_scan_range, current_time)

    def _scan_obstacles(self, mujoco_data: mujoco.MjData, robot_yaw: float) -> list[tuple[float, float]]:
        return [(angle, self._ray_distance(mujoco_data, robot_yaw, angle)) for angle in self._front_ray_angles]

    def get_robot_position(self, mujoco_data: mujoco.MjData) -> np.ndarray:
        return mujoco_data.xpos[self._base_id, :2].copy()

    def get_robot_yaw(self, mujoco_data: mujoco.MjData) -> float:
        return yaw_from_quat(mujoco_data.qpos[3:7])

    def _advance_path_target(self, robot_pos: np.ndarray) -> None:
        while self._path_target_index < len(self.path) - 1:
            distance = float(np.linalg.norm(self.get_current_target() - robot_pos))
            if distance > self._waypoint_radius:
                break
            self._path_target_index += 1

    def _update_visualization(self, robot_pos: np.ndarray, robot_yaw: float, current_time: float) -> None:
        if self.visualizer is None:
            return
        if current_time - self._last_visualization_time < self._visualization_interval:
            return
        self.visualizer.update(
            self.grid,
            robot_pos,
            robot_yaw,
            self.path,
            self._goal,
            self.trajectory[-self._visualization_trail_limit :],
            current_time,
        )
        self._last_visualization_time = current_time

    def _ray_distance(self, data: mujoco.MjData, current_yaw: float, local_angle: float) -> float:
        self._ray_origin[:] = data.xpos[self._base_id]
        self._ray_origin[0] += 0.35 * math.cos(current_yaw)
        self._ray_origin[1] += 0.35 * math.sin(current_yaw)
        self._ray_origin[2] = 0.32
        world_angle = current_yaw + local_angle
        self._ray_direction[:] = (math.cos(world_angle), math.sin(world_angle), 0.0)
        self._ray_geomid[0] = -1
        distance = mujoco.mj_ray(
            self._model,
            data,
            self._ray_origin,
            self._ray_direction,
            None,
            1,
            self._base_id,
            self._ray_geomid,
        )
        if distance < 0.0:
            return self._max_scan_range
        return min(float(distance), self._max_scan_range)

    def _apply_policy(self, data: mujoco.MjData, command: np.ndarray) -> None:
        if data.time >= self._next_policy_time - 1e-9:
            obs = self._observation(data, command)
            action_batch = cast("np.ndarray", self._session.run([self._output_name], {self._input_name: obs})[0])
            action = action_batch[0]
            self._last_action[:] = action.astype(np.float32)
            self._target_joint_pos[:] = self._last_action * self._action_scale + self._action_offset
            self._next_policy_time += self._policy_dt
        self._apply_pd_torques(data)

    def _observation(self, data: mujoco.MjData, command: np.ndarray) -> np.ndarray:
        obs = self._obs[0]
        obs[0:3] = data.sensordata[self._gyro_sensor_adr : self._gyro_sensor_adr + 3]
        obs[3:6] = quat_conjugate_rotate(data.qpos[3:7], self._gravity_world)
        obs[6:9] = command
        obs[9:21] = data.qpos[self._qpos_adr] - self._default_joint_pos
        obs[21:33] = data.qvel[self._qvel_adr]
        obs[33:45] = self._last_action
        return self._obs

    def _apply_pd_torques(self, data: mujoco.MjData) -> None:
        joint_pos = data.qpos[self._qpos_adr]
        joint_vel = data.qvel[self._qvel_adr]
        torques = self._stiffness * (self._target_joint_pos - joint_pos) - self._damping * joint_vel
        data.ctrl[:] = 0.0
        for policy_i, actuator_i in enumerate(self._joint_map):
            data.ctrl[actuator_i] = float(
                np.clip(torques[policy_i], self._ctrl_low[actuator_i], self._ctrl_high[actuator_i])
            )

    def _navigation_status(self, data: mujoco.MjData) -> ControllerStatus:
        robot_pos = self.get_robot_position(data)
        distance = float(np.linalg.norm(self._goal - robot_pos))
        desired_yaw = math.atan2(float(self._goal[1] - robot_pos[1]), float(self._goal[0] - robot_pos[0]))
        yaw_error = wrap_angle(desired_yaw - self.get_robot_yaw(data))
        base_position = data.xpos[self._base_id]
        mode = "nav" if self.path else "planning"
        return ControllerStatus(
            mode=mode,
            target_name="goal",
            distance=distance,
            yaw_error=yaw_error,
            base_position=(float(base_position[0]), float(base_position[1]), float(base_position[2])),
        )

    @staticmethod
    def _status_with_mode(status: ControllerStatus, mode: str) -> ControllerStatus:
        return ControllerStatus(
            mode=mode,
            target_name=status.target_name,
            distance=status.distance,
            yaw_error=status.yaw_error,
            base_position=status.base_position,
        )

    @staticmethod
    def _sensor_adr(model: mujoco.MjModel, names: tuple[str, ...]) -> int:
        for name in names:
            sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sensor_id >= 0:
                return int(model.sensor_adr[sensor_id])
        raise ValueError(f"Required gyro sensor not found. Tried: {names}")
