import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if TYPE_CHECKING:
    from collections.abc import Iterable

    from controllers.common import Controller

ARTIFACTS = ROOT / "artifacts"
VALIDATION_ENV = "MUJOCO_LABS_VALIDATE"
XML_DIR = ROOT / "vendor" / "unitree_rl_mjlab" / "src" / "assets" / "robots" / "unitree_go2" / "xmls"
BARE_SCENE = XML_DIR / "scene_go2.xml"
LAB2_SCENE = XML_DIR / "lab2_editing_scene.xml"
TILTED_SCENE = XML_DIR / "tilted_stand_scene.xml"
LINE_ROUTE_SCENE = XML_DIR / "line_route_scene.xml"
ROUTE_ONLY_SCENE = XML_DIR / "route_only_scene.xml"
ROUTE_WALL_SCENE = XML_DIR / "route_wall_scene.xml"
ROUTE_RAMP_SCENE = XML_DIR / "route_ramp_scene.xml"
FULL_ROUTE_SCENE = XML_DIR / "route_scene.xml"
COLOR_WALLS_SCENE = XML_DIR / "lab8_color_walls_scene.xml"
# LAB12_SCENE = XML_DIR / "map01_s_curve.xml"
# LAB12_SCENE = XML_DIR / "map02_funnel.xml"
# LAB12_SCENE = XML_DIR / "map03_corridor_maze.xml" # diagonal
LAB12_SCENE = XML_DIR / "map04_islands.xml"
# LAB12_SCENE = XML_DIR / "map05_dead_ends.xml"
# LAB12_SCENE = XML_DIR / "map06_narrow_gaps.xml"
# LAB12_SCENE = XML_DIR / "map07_u_turn.xml"
# LAB12_SCENE = XML_DIR / "map08_diagonal_slalom.xml"
# LAB12_SCENE = XML_DIR / "map09_chicane.xml"
# LAB12_SCENE = XML_DIR / "map10_complex_labyrinth.xml"
# LAB12_SCENE = XML_DIR / "lab12_scene.xml"
DEFAULT_CAMERA = "free"
REALTIME = True
MAX_STEPS_PER_FRAME = 25


class ViewerCamera(Protocol):
    type: object
    fixedcamid: int
    lookat: np.ndarray
    distance: float
    azimuth: float
    elevation: float


class ViewerHandle(Protocol):
    cam: ViewerCamera

    def is_running(self) -> bool: ...
    def set_texts(self, texts: tuple[object, ...]) -> None: ...
    def sync(self) -> None: ...


def artifact_dir(name: str) -> Path:
    path = ARTIFACTS / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_model_from_path(scene_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData]:
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file does not exist: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    reset_to_home(model, data)
    return model, data


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise ValueError("The robot model must define a 'home' keyframe.")

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.ctrl[:] = model.key_ctrl[key_id]
    mujoco.mj_forward(model, data)


def load_bare_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(BARE_SCENE)


def load_lab2_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(LAB2_SCENE)


def load_tilted_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(TILTED_SCENE)


def load_line_route_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(LINE_ROUTE_SCENE)


def load_route_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(ROUTE_ONLY_SCENE)


def load_wall_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(ROUTE_WALL_SCENE)


def load_ramp_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(ROUTE_RAMP_SCENE)


def load_full_route_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(FULL_ROUTE_SCENE)


def load_lab12_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_model_from_path(LAB12_SCENE)


def load_default_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    return load_full_route_model()


def validation_mode() -> bool:
    return os.environ.get(VALIDATION_ENV, "").lower() in {"1", "true", "yes", "on"}


def open_lab_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Controller,
    title: str,
    *,
    duration_s: float = 10.0,
) -> None:
    if validation_mode():
        return

    print(f"Открывается визуализатор MuJoCo: {title}")
    print(f"Рекомендуемое время наблюдения: {duration_s:.0f} с.")
    print("Можно вращать камеру мышью. Чтобы продолжить, закройте окно визуализатора.")
    run_viewer(
        model,
        data,
        controller,
        DEFAULT_CAMERA,
        REALTIME,
        MAX_STEPS_PER_FRAME,
        title=title,
        max_time=None,
    )


def configure_viewer_camera(model: mujoco.MjModel, viewer: ViewerHandle, camera_name: str) -> None:
    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = (1.5, 1.0, 0.35)
        viewer.cam.distance = 4.8
        viewer.cam.azimuth = -135.0
        viewer.cam.elevation = -25.0
        return

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Unknown camera: {camera_name}")

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Controller,
    camera_name: str,
    realtime: bool,
    max_steps_per_frame: int,
    *,
    title: str = "Controller",
    max_time: float | None = None,
) -> None:
    import mujoco.viewer  # noqa: PLC0415

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer_handle = cast("ViewerHandle", viewer)
        configure_viewer_camera(model, viewer_handle, camera_name)
        viewer_handle.sync()

        status = controller.step(data)
        wall_start = time.perf_counter()
        sim_start = data.time

        while viewer_handle.is_running():
            if max_time is not None and data.time - sim_start >= max_time:
                break

            if realtime:
                target_sim_time = sim_start + (time.perf_counter() - wall_start)
                step_count = 0
                while data.time < target_sim_time and step_count < max_steps_per_frame:
                    status = controller.step(data)
                    mujoco.mj_step(model, data)
                    step_count += 1

                if step_count == max_steps_per_frame and data.time < target_sim_time:
                    wall_start = time.perf_counter()
                    sim_start = data.time
            else:
                for _ in range(max_steps_per_frame):
                    status = controller.step(data)
                    mujoco.mj_step(model, data)

            viewer_handle.set_texts(
                (
                    mujoco.mjtFontScale.mjFONTSCALE_150,
                    mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    title,
                    (
                        f"{status.mode} | target={status.target_name} "
                        f"| dist={status.distance:.2f} "
                        f"| xyz=({status.base_position[0]:.2f}, "
                        f"{status.base_position[1]:.2f}, {status.base_position[2]:.2f})"
                    ),
                )
            )
            viewer_handle.sync()


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def csv_float(value: object) -> float:
    if isinstance(value, int | float | str):
        return float(value)
    raise TypeError(f"Expected numeric CSV value, got {type(value).__name__}.")


def csv_int(value: object) -> int:
    if isinstance(value, int | float | str):
        return int(value)
    raise TypeError(f"Expected integer CSV value, got {type(value).__name__}.")


def name_or_empty(model: mujoco.MjModel, obj_type: mujoco.mjtObj, index: int) -> str:
    return mujoco.mj_id2name(model, obj_type, index) or ""


def object_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    value = mujoco.mj_name2id(model, obj_type, name)
    if value < 0:
        raise ValueError(f"Object not found: {name}")
    return int(value)


def route_sites(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    names: tuple[str, ...] = ("wp_a", "wp_b", "wp_c", "wp_d"),
) -> list[tuple[str, float, float]]:
    mujoco.mj_forward(model, data)
    result = []
    for name in names:
        site_id = object_id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        x, y = data.site_xpos[site_id, :2]
        result.append((name, float(x), float(y)))
    return result


def yaw_from_quat(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def roll_pitch_yaw(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    yaw = yaw_from_quat(quat)
    return roll, pitch, yaw


def quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)], dtype=float)


def body_position(model: mujoco.MjModel, data: mujoco.MjData, body_name: str = "base_link") -> np.ndarray:
    body_id = object_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def step_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Controller,
    steps: int,
    sample_every: int = 50,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for step in range(steps):
        status = controller.step(data)
        mujoco.mj_step(model, data)
        if step % sample_every == 0 or step == steps - 1:
            roll, pitch, yaw = roll_pitch_yaw(data.qpos[3:7])
            rows.append(
                {
                    "step": step,
                    "time": round(float(data.time), 4),
                    "mode": status.mode,
                    "target": status.target_name,
                    "target_distance": round(float(status.distance), 4),
                    "base_x": round(float(status.base_position[0]), 4),
                    "base_y": round(float(status.base_position[1]), 4),
                    "base_z": round(float(status.base_position[2]), 4),
                    "roll": round(roll, 4),
                    "pitch": round(pitch, 4),
                    "yaw": round(yaw, 4),
                }
            )
    return rows


def ray_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    origin: tuple[float, float, float],
    yaw: float,
    local_angle: float,
    exclude_body: int = -1,
) -> tuple[float, str]:
    direction = np.array([math.cos(yaw + local_angle), math.sin(yaw + local_angle), 0.0], dtype=float)
    geom_id = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(
        model,
        data,
        np.array(origin, dtype=float),
        direction,
        None,
        1,
        exclude_body,
        geom_id,
    )
    if distance < 0.0:
        return math.inf, ""
    geom_name = name_or_empty(model, mujoco.mjtObj.mjOBJ_GEOM, int(geom_id[0]))
    return float(distance), geom_name


def summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    if not rows:
        return {}
    base_z = [csv_float(row["base_z"]) for row in rows]
    return {
        "samples": len(rows),
        "last_mode": rows[-1]["mode"],
        "last_target": rows[-1]["target"],
        "last_time": rows[-1]["time"],
        "min_base_z": round(min(base_z), 4),
        "max_base_z": round(max(base_z), 4),
    }
