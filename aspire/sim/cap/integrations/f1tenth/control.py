# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Privileged control and observation API for F1TENTH."""

from __future__ import annotations

from typing import Any

import numpy as np

from aspire.sim.cap.envs.base import BaseEnv
from aspire.sim.cap.integrations.base_api import ApiBase


class F1TenthControlApi(ApiBase):
    """Full-state F1TENTH API used by the initial native integration."""

    def __init__(self, env: BaseEnv) -> None:
        super().__init__(env)
        required = ("get_observation", "get_reference_path", "step")
        if not all(hasattr(env, name) for name in required):
            raise TypeError("F1TenthControlApi requires an F1TENTH low-level environment")

    def functions(self) -> dict[str, Any]:
        return {
            "get_observation": self.get_observation,
            "get_reference_path": self.get_reference_path,
            "drive": self.drive,
            "stop": self.stop,
            "render_overhead": self.render_overhead,
        }

    def get_observation(self) -> dict[str, Any]:
        """Return all available vehicle and simulator telemetry.

        Returns:
            Dictionary containing LiDAR, global pose, velocity, collision state,
            lap progress, previous action, timing, and the raw simulator observation.
        """
        return self._env.get_observation()

    def get_reference_path(self) -> np.ndarray:
        """Return the active track centerline as an ``(N, 2)`` array."""
        return self._env.get_reference_path()

    def drive(
        self,
        steering_angle: float,
        speed: float,
        steps: int = 1,
    ) -> dict[str, Any]:
        """Apply a steering and speed command for one or more control periods.

        Args:
            steering_angle: Front-wheel steering angle in radians, clipped to
                ``[-0.4189, 0.4189]``.
            speed: Desired forward speed in meters per second, clipped to ``[0, 8]``.
            steps: Number of control periods to repeat the command.

        Returns:
            Full observation after the final control period. Execution stops
            early when the lap completes, a collision occurs, or time expires.
        """
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        observation: dict[str, Any] = self._env.get_observation()
        for _ in range(steps):
            observation, _, terminated, truncated, _ = self._env.step(
                {"steering_angle": steering_angle, "speed": speed}
            )
            if terminated or truncated:
                break
        return observation

    def stop(self) -> dict[str, Any]:
        """Command zero speed for one control period and return the observation."""
        return self.drive(0.0, 0.0)

    def render_overhead(self) -> np.ndarray:
        """Return the current privileged overhead RGB visualization."""
        return self._env.render(mode="rgb_array")


__all__ = ["F1TenthControlApi"]
