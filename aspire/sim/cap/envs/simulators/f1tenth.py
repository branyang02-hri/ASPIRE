# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ASPIRE wrapper for the pinned F1TENTH Gym simulator."""

from __future__ import annotations

import csv
import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from gymnasium import spaces
from PIL import Image, ImageDraw

from aspire.sim.cap.envs.base import BaseEnv

CANONICAL_LEVINE_PNG_SHA256 = "f5983fdc8e1a2395f533502a582089d0c0b76835523f4d21a00982da4af2b29e"
CANONICAL_LEVINE_YAML_SHA256 = "6c422686da9fdc28ace3613aae31520102225516c2f09aea51b6aedce383c1fe"
F1TENTH_EFFECTIVE_LEVINE_SHA256 = "234b5bcc50ea02fd3845897f88bbc6066984f200ffb1ed664734d19015b0a315"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_levine_assets() -> tuple[Path, Path]:
    sim_root = Path(__file__).resolve().parents[3]
    base = sim_root / "assets" / "f1tenth" / "levine" / "levine"
    return base, base.parent / "levine_reference_path.csv"


@dataclass(frozen=True)
class LapProgress:
    completed: bool
    progress_fraction: float
    ordered_gate_fraction: float
    cross_track_error: float
    direction: int


class OrderedLapTracker:
    """Track continuous progress around a cyclic path in either direction."""

    def __init__(
        self,
        path: np.ndarray,
        *,
        gate_count: int = 24,
        start_radius: float = 1.0,
        departure_radius: float = 2.0,
        max_cross_track: float = 1.4,
        max_step_progress: float = 1.0,
    ) -> None:
        self.path = np.asarray(path, dtype=np.float64)
        if self.path.ndim != 2 or self.path.shape[0] < 8 or self.path.shape[1] != 2:
            raise ValueError("reference path must have shape (N, 2) with at least eight points")
        self.gate_count = int(gate_count)
        self.start_radius = float(start_radius)
        self.departure_radius = float(departure_radius)
        self.max_cross_track = float(max_cross_track)
        self.max_step_progress = float(max_step_progress)
        segments = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.arc = np.concatenate(([0.0], np.cumsum(segments)))
        self.length = float(self.arc[-1] + np.linalg.norm(self.path[-1] - self.path[0]))
        self.reset()

    def reset(self) -> None:
        self.start_xy: np.ndarray | None = None
        self.previous_s: float | None = None
        self.net_progress = 0.0
        self.direction = 0
        self.departed = False
        self.next_gate = 1
        self.completed = False

    def update(self, x: float, y: float) -> LapProgress:
        xy = np.array([x, y], dtype=np.float64)
        if self.start_xy is None:
            self.start_xy = xy.copy()
        distances = np.linalg.norm(self.path - xy, axis=1)
        nearest = int(np.argmin(distances))
        s_value = float(self.arc[nearest])
        cross_track = float(distances[nearest])

        if self.previous_s is not None:
            delta = s_value - self.previous_s
            if delta > self.length / 2.0:
                delta -= self.length
            elif delta < -self.length / 2.0:
                delta += self.length
            if abs(delta) <= self.max_step_progress and cross_track <= self.max_cross_track:
                self.net_progress += delta
        self.previous_s = s_value

        assert self.start_xy is not None
        start_distance = float(np.linalg.norm(xy - self.start_xy))
        if start_distance >= self.departure_radius:
            self.departed = True
        if self.direction == 0 and abs(self.net_progress) >= 2.0:
            self.direction = 1 if self.net_progress > 0.0 else -1

        directed = self.direction * self.net_progress if self.direction else abs(self.net_progress)
        directed = max(0.0, directed)
        while self.next_gate < self.gate_count:
            threshold = self.next_gate * self.length / self.gate_count
            if directed + 0.3 < threshold:
                break
            self.next_gate += 1

        returned = start_distance <= self.start_radius
        if (
            self.departed
            and self.direction != 0
            and self.next_gate >= self.gate_count
            and directed >= 0.90 * self.length
            and returned
            and cross_track <= self.max_cross_track
        ):
            self.completed = True

        return LapProgress(
            completed=self.completed,
            progress_fraction=min(1.0, directed / self.length),
            ordered_gate_fraction=self.next_gate / self.gate_count,
            cross_track_error=cross_track,
            direction=self.direction,
        )


class F1TenthLevineLowLevel(BaseEnv):
    """F1TENTH Levine environment with full simulator telemetry exposed."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        map_path: str | None = None,
        reference_path: str | None = None,
        max_steps: int = 18_000,
        control_repeat: int = 5,
        seed: int = 101,
        start_poses: list[list[float]] | None = None,
        privileged: bool = True,
        enable_render: bool = True,
        viser_debug: bool = False,
        render_size: int = 384,
        video_stride: int = 4,
    ) -> None:
        super().__init__()
        canonical_map, canonical_reference = canonical_levine_assets()
        self.map_base = Path(map_path).expanduser().resolve() if map_path else canonical_map
        if self.map_base.suffix in {".png", ".yaml"}:
            self.map_base = self.map_base.with_suffix("")
        self.reference_path_file = (
            Path(reference_path).expanduser().resolve() if reference_path else canonical_reference
        )
        self._validate_assets(require_canonical=map_path is None)

        self.max_steps = int(max_steps)
        self.control_repeat = int(control_repeat)
        self.default_seed = int(seed)
        self.start_poses = start_poses or [
            [0.0, 0.0, 0.0],
            [0.0, 0.05, 0.015],
            [0.0, -0.05, -0.015],
        ]
        self.privileged = bool(privileged)
        self.enable_render = bool(enable_render)
        self.viser_debug = bool(viser_debug)
        self.render_size = int(render_size)
        self.video_stride = max(1, int(video_stride))

        self.reference_path = self._load_reference_path(self.reference_path_file)
        self._lap_tracker = OrderedLapTracker(self.reference_path)
        self._lap = LapProgress(False, 0.0, 0.0, math.inf, 0)
        self._env: Any | None = None
        self._raw_obs: dict[str, Any] | None = None
        self._step_count = 0
        self._sim_step_count = 0
        self._sim_time = 0.0
        self._collision = False
        self._terminated = False
        self._truncated = False
        self._last_action = np.zeros(2, dtype=np.float64)
        self._trajectory: list[tuple[float, float, float]] = []
        self._record_frames = False
        self._frame_buffer: list[np.ndarray] = []
        self._map_image, self._map_metadata = self._load_map_for_rendering()
        self._render_crop = self._compute_render_crop()

        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.0], dtype=np.float32),
            high=np.array([0.4189, 8.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict({})

    def _validate_assets(self, *, require_canonical: bool) -> None:
        png = self.map_base.with_suffix(".png")
        yaml_path = self.map_base.with_suffix(".yaml")
        missing = [
            str(path) for path in (png, yaml_path, self.reference_path_file) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(f"missing F1TENTH assets: {', '.join(missing)}")
        if require_canonical:
            hashes = (_sha256(png), _sha256(yaml_path))
            expected = (CANONICAL_LEVINE_PNG_SHA256, CANONICAL_LEVINE_YAML_SHA256)
            if hashes != expected:
                raise RuntimeError("canonical Levine asset hash mismatch")

    @staticmethod
    def _load_reference_path(path: Path) -> np.ndarray:
        points = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                points.append((float(row["x"]), float(row["y"])))
        return np.asarray(points, dtype=np.float64)

    def _load_map_for_rendering(self) -> tuple[Image.Image, dict[str, Any]]:
        image = Image.open(self.map_base.with_suffix(".png")).convert("RGB")
        metadata = yaml.safe_load(self.map_base.with_suffix(".yaml").read_text())
        return image, metadata

    def _create_env(self, seed: int) -> Any:
        os.environ.setdefault("PYGLET_HEADLESS", "true")
        from f110_gym.envs.f110_env import F110Env

        return F110Env(
            map=str(self.map_base),
            map_ext=".png",
            num_agents=1,
            seed=seed,
            lidar_dist=0.275,
            timestep=0.01,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        actual_seed = self.default_seed if seed is None else int(seed)
        options = options or {}
        if "start_pose" in options:
            start_pose = np.asarray(options["start_pose"], dtype=np.float64)
        else:
            start_pose = np.asarray(
                self.start_poses[actual_seed % len(self.start_poses)], dtype=np.float64
            )
        if start_pose.shape != (3,):
            raise ValueError("start_pose must contain x, y, and heading")

        self.close()
        self._env = self._create_env(actual_seed)
        self._raw_obs, _, _, _ = self._env.reset(start_pose.reshape(1, 3))
        self._step_count = 0
        self._sim_step_count = 1
        self._sim_time = 0.01
        self._collision = bool(self._raw_obs["collisions"][0])
        self._terminated = False
        self._truncated = False
        self._last_action = np.zeros(2, dtype=np.float64)
        self._trajectory = []
        self._lap_tracker.reset()
        self._record_state()
        if self._record_frames:
            self._frame_buffer = [self.render()]
        else:
            self._frame_buffer = []
        return self.get_observation(), {"seed": actual_seed, "start_pose": start_pose.tolist()}

    def _record_state(self) -> None:
        assert self._raw_obs is not None
        x = float(self._raw_obs["poses_x"][0])
        y = float(self._raw_obs["poses_y"][0])
        theta = float(self._raw_obs["poses_theta"][0])
        self._lap = self._lap_tracker.update(x, y)
        self._trajectory.append((x, y, theta))

    @staticmethod
    def _parse_action(action: Any) -> np.ndarray:
        if isinstance(action, dict):
            value = np.array([action["steering_angle"], action["speed"]], dtype=np.float64)
        else:
            value = np.asarray(action, dtype=np.float64).reshape(-1)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("action must contain finite steering_angle and speed values")
        value[0] = np.clip(value[0], -0.4189, 0.4189)
        value[1] = np.clip(value[1], 0.0, 8.0)
        return value

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._env is None or self._raw_obs is None:
            raise RuntimeError("environment must be reset before stepping")
        if self._terminated or self._truncated:
            return (
                self.get_observation(),
                self.compute_reward(),
                self._terminated,
                self._truncated,
                {"reason": self._termination_reason()},
            )

        command = self._parse_action(action)
        self._last_action = command.copy()
        for _ in range(self.control_repeat):
            self._raw_obs, _, _, _ = self._env.step(command.reshape(1, 2))
            self._sim_step_count += 1
            self._sim_time += 0.01
            self._collision = bool(self._raw_obs["collisions"][0])
            self._record_state()
            if self._collision or self._lap.completed:
                self._terminated = True
                break
            if self._sim_step_count >= self.max_steps:
                self._truncated = True
                break
        self._step_count += 1
        if self._record_frames and self._step_count % self.video_stride == 0:
            self._frame_buffer.append(self.render())
        info = {
            "reason": self._termination_reason(),
            "completed_lap": self._lap.completed,
            "collision": self._collision,
            "progress_fraction": self._lap.progress_fraction,
        }
        return (
            self.get_observation(),
            self.compute_reward(),
            self._terminated,
            self._truncated,
            info,
        )

    def _termination_reason(self) -> str:
        if self._lap.completed:
            return "completed_lap"
        if self._collision:
            return "collision"
        if self._truncated:
            return "timeout"
        return "running"

    def get_observation(self) -> dict[str, Any]:
        if self._raw_obs is None:
            raise RuntimeError("environment has not been reset")
        raw = {
            key: np.asarray(value).copy() if isinstance(value, (list, tuple, np.ndarray)) else value
            for key, value in self._raw_obs.items()
        }
        ranges = np.asarray(self._raw_obs["scans"][0], dtype=np.float64).copy()
        return {
            "scan": {
                "ranges": ranges,
                "angle_min": -2.35,
                "angle_increment": 4.7 / max(1, len(ranges) - 1),
                "range_min": 0.0,
                "range_max": 30.0,
                "stamp": self._sim_time,
            },
            "pose": np.array(
                [
                    self._raw_obs["poses_x"][0],
                    self._raw_obs["poses_y"][0],
                    self._raw_obs["poses_theta"][0],
                ],
                dtype=np.float64,
            ),
            "velocity": {
                "x": float(self._raw_obs["linear_vels_x"][0]),
                "y": float(self._raw_obs["linear_vels_y"][0]),
                "yaw_rate": float(self._raw_obs["ang_vels_z"][0]),
            },
            "collision": self._collision,
            "lap": {
                "completed": self._lap.completed,
                "progress_fraction": self._lap.progress_fraction,
                "ordered_gate_fraction": self._lap.ordered_gate_fraction,
                "cross_track_error": self._lap.cross_track_error,
                "direction": self._lap.direction,
            },
            "last_action": self._last_action.copy(),
            "sim_time": self._sim_time,
            "step_count": self._step_count,
            "done": self._terminated or self._truncated,
            "termination_reason": self._termination_reason(),
            "raw_observation": raw,
        }

    def get_reference_path(self) -> np.ndarray:
        return self.reference_path.copy()

    def compute_reward(self) -> float:
        if self._lap.completed and not self._collision:
            return 1.0
        reward = 0.72 * self._lap.progress_fraction + 0.28 * self._lap.ordered_gate_fraction
        if self._collision:
            reward -= 0.15
        return float(np.clip(reward, 0.0, 0.99))

    def task_completed(self) -> bool:
        return bool(self._lap.completed and not self._collision)

    def _world_to_pixel(self, xy: np.ndarray) -> list[tuple[int, int]]:
        resolution = float(self._map_metadata["resolution"])
        origin_x, origin_y = map(float, self._map_metadata["origin"][:2])
        height = self._map_image.height
        pixels = []
        for x, y in np.asarray(xy):
            px = int(round((float(x) - origin_x) / resolution))
            py = height - 1 - int(round((float(y) - origin_y) / resolution))
            pixels.append((px, py))
        return pixels

    def _compute_render_crop(self) -> tuple[int, int, int, int]:
        pixels = np.asarray(self._world_to_pixel(self.reference_path), dtype=np.int64)
        margin = 160
        min_x, max_x = int(pixels[:, 0].min()), int(pixels[:, 0].max())
        min_y, max_y = int(pixels[:, 1].min()), int(pixels[:, 1].max())
        side = min(
            max(max_x - min_x, max_y - min_y) + 2 * margin,
            self._map_image.width,
            self._map_image.height,
        )
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        left = min(max(0, center_x - side // 2), self._map_image.width - side)
        top = min(max(0, center_y - side // 2), self._map_image.height - side)
        right = left + side
        bottom = top + side
        return left, top, right, bottom

    def render(self, mode: str = "rgb_array") -> np.ndarray:
        if mode != "rgb_array":
            raise ValueError("F1TENTH integration supports only rgb_array rendering")
        canvas = self._map_image.copy()
        draw = ImageDraw.Draw(canvas)
        draw.line(self._world_to_pixel(self.reference_path), fill=(40, 150, 220), width=4)
        if self._trajectory:
            trajectory = np.asarray([(x, y) for x, y, _ in self._trajectory])
            draw.line(self._world_to_pixel(trajectory), fill=(220, 55, 45), width=5)
            x, y, theta = self._trajectory[-1]
            center = np.asarray(self._world_to_pixel(np.array([[x, y]]))[0], dtype=np.float64)
            scale = 18.0
            forward = np.array([math.cos(theta), -math.sin(theta)]) * scale
            left = np.array([-forward[1], forward[0]]) * 0.55
            polygon = [
                tuple(center + forward),
                tuple(center - forward + left),
                tuple(center - forward - left),
            ]
            draw.polygon(polygon, fill=(20, 40, 220))
        canvas = canvas.crop(self._render_crop)
        canvas.thumbnail((self.render_size, self.render_size), Image.Resampling.LANCZOS)
        return np.asarray(canvas, dtype=np.uint8)

    def enable_video_capture(
        self,
        enabled: bool = True,
        *,
        clear: bool = True,
        wrist_camera: bool = False,
    ) -> None:
        del wrist_camera
        self._record_frames = bool(enabled)
        if clear:
            self._frame_buffer = []

    def get_video_frames(self, *, clear: bool = False) -> list[np.ndarray]:
        frames = [frame.copy() for frame in self._frame_buffer]
        if clear:
            self._frame_buffer = []
        return frames

    def get_video_frame_count(self) -> int:
        return len(self._frame_buffer)

    def get_video_frames_range(self, start: int, end: int) -> list[np.ndarray]:
        return [frame.copy() for frame in self._frame_buffer[start:end]]

    def close(self) -> None:
        if self._env is not None:
            close = getattr(self._env, "close", None)
            if callable(close):
                close()
        self._env = None


__all__ = [
    "CANONICAL_LEVINE_PNG_SHA256",
    "CANONICAL_LEVINE_YAML_SHA256",
    "F1TENTH_EFFECTIVE_LEVINE_SHA256",
    "F1TenthLevineLowLevel",
    "LapProgress",
    "OrderedLapTracker",
    "canonical_levine_assets",
]
