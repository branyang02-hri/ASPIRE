# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restricted LiDAR-only code-execution task for Spielberg."""

from aspire.sim.cap.envs.tasks.f1tenth.levine_lidar_only import (
    F1TenthLidarOnlyCodeEnv,
)

SPIELBERG_PROMPT = """
You are controlling an F1TENTH race car in simulation.
Goal: complete one full lap of the Spielberg course without colliding.
Control must be reactive and local. The controller receives only LiDAR geometry and
ranges, longitudinal speed, yaw rate, timestamp, and episode termination. It has no
map, reference path, global pose, progress, reward, overhead view, or simulator object.
Write executable Python code only, without Markdown fences. The API functions below
are already imported. Import numpy or standard-library modules explicitly when needed.
"""


class F1TenthSpielbergLidarOnlyCodeEnv(F1TenthLidarOnlyCodeEnv):
    """Spielberg evaluator with isolated sensor-only controller execution."""

    prompt = SPIELBERG_PROMPT


__all__ = ["F1TenthSpielbergLidarOnlyCodeEnv"]
