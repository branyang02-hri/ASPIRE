# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sensor-only F1TENTH API; this object remains in the evaluator process."""

from __future__ import annotations

from typing import Any

import numpy as np

from aspire.sim.cap.envs.base import BaseEnv
from aspire.sim.cap.integrations.base_api import ApiBase


class F1TenthLidarApi(ApiBase):
    """Expose only local LiDAR/motion sensing and vehicle actuation."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        if not hasattr(env, "step") or not hasattr(env, "_raw_obs"):
            raise TypeError("F1TenthLidarApi requires an F1TENTH low-level environment")

    def functions(self) -> dict[str, Any]:
        return {
            "get_scan": self.get_scan,
            "drive": self.drive,
            "stop": self.stop,
        }

    def get_scan(self) -> dict[str, Any]:
        """Return only LiDAR geometry, local motion estimates, time, and termination.

        Returns:
            A dictionary with exactly ``scan``, ``speed``, ``timestamp``, and ``done``.
            ``scan`` contains ranges in meters plus angle/range geometry. ``speed``
            contains longitudinal wheel speed and local IMU yaw rate estimates.
        """
        raw = self._env._raw_obs
        if raw is None:
            raise RuntimeError("environment has not been reset")
        ranges = np.asarray(raw["scans"][0], dtype=np.float64)
        return {
            "scan": {
                "ranges": ranges.tolist(),
                "angle_min": -2.35,
                "angle_increment": 4.7 / max(1, len(ranges) - 1),
                "range_min": 0.0,
                "range_max": 30.0,
            },
            "speed": {
                "longitudinal": float(raw["linear_vels_x"][0]),
                "yaw_rate": float(raw["ang_vels_z"][0]),
            },
            "timestamp": float(self._env._sim_time),
            "done": bool(self._env._terminated or self._env._truncated),
        }

    def drive(
        self,
        steering_angle: float,
        speed: float,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Apply a steering/speed command and return the next local sensor packet.

        Args:
            steering_angle: Front-wheel angle in radians, clipped to [-0.4189, 0.4189].
            speed: Desired forward speed in meters per second, clipped to [0, 8].
            steps: Control periods to repeat, from 1 through 50.

        Returns:
            The same restricted packet returned by ``get_scan``.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 50:
            raise ValueError("steps must be an integer from 1 through 50")
        if not np.isfinite(float(steering_angle)) or not np.isfinite(float(speed)):
            raise ValueError("steering_angle and speed must be finite")
        for _ in range(steps):
            _, _, terminated, truncated, _ = self._env.step(
                {"steering_angle": float(steering_angle), "speed": float(speed)}
            )
            if terminated or truncated:
                break
        return self.get_scan()

    def stop(self) -> dict[str, Any]:
        """Command zero speed for one control period and return local sensors."""
        return self.drive(0.0, 0.0)


__all__ = ["F1TenthLidarApi"]
