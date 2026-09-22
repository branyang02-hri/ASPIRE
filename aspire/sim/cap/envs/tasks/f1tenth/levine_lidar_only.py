# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Restricted LiDAR-only code-execution task for Levine."""

from __future__ import annotations

from typing import Any

from aspire.sim.cap.envs.base import ObsType
from aspire.sim.cap.envs.tasks.base import CodeExecutionEnvBase
from aspire.sim.cap.envs.tasks.f1tenth.restricted_executor import (
    RestrictedControllerExecutor,
)

LEVINE_PROMPT = """
You are controlling an F1TENTH race car in simulation.
Goal: complete one full lap of the Levine course without colliding.
Control must be reactive and local. The controller receives only LiDAR geometry and
ranges, longitudinal speed, yaw rate, timestamp, and episode termination. It has no
map, reference path, global pose, progress, reward, overhead view, or simulator object.
Write executable Python code only, without Markdown fences. The API functions below
are already imported. Import numpy or standard-library modules explicitly when needed.
"""


class F1TenthLidarOnlyCodeEnv(CodeExecutionEnvBase):
    """Base evaluator with isolated sensor-only controller execution."""

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self._restricted_executor = RestrictedControllerExecutor()

    def _init_exec_globals(self) -> None:
        self._exec_globals = {"RESULT": None}

    def _lidar_api(self) -> Any:
        if len(self._apis) != 1:
            raise RuntimeError("LiDAR-only task requires exactly one API")
        return next(iter(self._apis.values()))

    def _get_observation(self) -> dict[str, Any]:
        return self._lidar_api().get_scan()

    def _handle_restricted_request(self, method: str, params: dict[str, Any]) -> Any:
        api = self._lidar_api()
        allowed = api.functions()
        if method not in allowed:
            raise PermissionError(f"API method is not available: {method}")
        return allowed[method](**params)

    def _exec_user_code(self, code: str) -> dict[str, Any]:
        result = self._restricted_executor.run(code, self._handle_restricted_request)
        self._exec_globals["RESULT"] = result.get("result")
        return result

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self._step_count = 0
        _, privileged_info = self.low_level_env.reset(seed=seed, options=options)
        self._init_exec_globals()
        public_info = {"task_prompt": self._task_prompt}
        if "seed" in privileged_info:
            public_info["seed"] = privileged_info["seed"]
        return self._get_observation(), public_info


class F1TenthLevineLidarOnlyCodeEnv(F1TenthLidarOnlyCodeEnv):
    """Levine evaluator with isolated sensor-only controller execution."""

    prompt = LEVINE_PROMPT


__all__ = ["F1TenthLidarOnlyCodeEnv", "F1TenthLevineLidarOnlyCodeEnv"]
