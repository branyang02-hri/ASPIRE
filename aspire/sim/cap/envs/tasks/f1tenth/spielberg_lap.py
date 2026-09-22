# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Code-execution task for completing one lap of Spielberg."""

from aspire.sim.cap.envs.tasks.base import CodeExecutionEnvBase

PROMPT = """
You are controlling an F1TENTH race car in simulation.
Goal: complete one full lap of the Spielberg course without colliding.
This integration is privileged: the API exposes LiDAR, global vehicle state,
lap metrics, an overhead image, and the reference path. Use any exposed information.
Write executable Python code only, without Markdown fences. The API functions below
are already imported. Import numpy or standard-library modules explicitly when needed.
"""


class F1TenthSpielbergCodeEnv(CodeExecutionEnvBase):
    """High-level ASPIRE environment for privileged Spielberg lap control."""

    prompt = PROMPT


__all__ = ["F1TenthSpielbergCodeEnv"]
