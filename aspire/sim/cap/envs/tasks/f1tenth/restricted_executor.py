# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OS-isolated execution for sensor-only F1TENTH controllers."""

from __future__ import annotations

import json
import resource
import select
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

_CHILD_RUNNER = r'''
import contextlib
import io
import json
import site
import sys
import traceback

site.addsitedir("/venv/lib/python{python_version}/site-packages")

_wire_in = sys.stdin
_wire_out = sys.stdout
_next_id = 0


class _CappedIO(io.StringIO):
    def __init__(self, limit=131072):
        super().__init__()
        self.limit = limit
        self.written = 0

    def write(self, value):
        remaining = max(0, self.limit - self.written)
        chunk = value[:remaining]
        self.written += len(chunk)
        return super().write(chunk)


def _request(method, **params):
    global _next_id
    _next_id += 1
    message = {{"type": "request", "id": _next_id, "method": method, "params": params}}
    _wire_out.write(json.dumps(message, separators=(",", ":")) + "\n")
    _wire_out.flush()
    response_line = _wire_in.readline()
    if not response_line:
        raise RuntimeError("sensor/control channel closed")
    response = json.loads(response_line)
    if response.get("id") != _next_id:
        raise RuntimeError("sensor/control channel response mismatch")
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "sensor/control request failed"))
    return response.get("result")


def get_scan():
    """Return the current LiDAR scan, local motion estimates, timestamp, and done flag."""
    return _request("get_scan")


def drive(steering_angle, speed, steps=1):
    """Apply a bounded steering/speed command and return the next sensor packet."""
    return _request(
        "drive", steering_angle=steering_angle, speed=speed, steps=steps
    )


def stop():
    """Command zero speed for one control period and return the next sensor packet."""
    return _request("stop")


initial = json.loads(_wire_in.readline())
code = initial["code"]
user_stdout = _CappedIO()
user_stderr = _CappedIO()
user_globals = {{
    "__name__": "__main__",
    "get_scan": get_scan,
    "drive": drive,
    "stop": stop,
    "RESULT": None,
}}
ok = True
with contextlib.redirect_stdout(user_stdout), contextlib.redirect_stderr(user_stderr):
    try:
        exec(compile(code, "<candidate_controller>", "exec"), user_globals, user_globals)
    except BaseException:
        ok = False
        traceback.print_exc()

result = user_globals.get("RESULT")
try:
    json.dumps(result)
except (TypeError, ValueError):
    result = repr(result)
finished = {{
    "type": "finished",
    "ok": ok,
    "stdout": user_stdout.getvalue(),
    "stderr": user_stderr.getvalue(),
    "result": result,
}}
_wire_out.write(json.dumps(finished, separators=(",", ":")) + "\n")
_wire_out.flush()
'''


class RestrictedControllerExecutor:
    """Run candidate code behind a narrow JSON-RPC channel inside bubblewrap."""

    def __init__(
        self,
        *,
        timeout_seconds: float = 600.0,
        max_requests: int = 25_000,
        max_line_bytes: int = 2_000_000,
    ) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.max_requests = int(max_requests)
        self.max_line_bytes = int(max_line_bytes)

    @staticmethod
    def _set_limits() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (600, 605))
        resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
        resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    @staticmethod
    def _sandbox_command() -> list[str]:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError("bubblewrap is required for restricted controller execution")

        venv = Path(sys.prefix).resolve()
        base = Path(sys.base_prefix).resolve()
        executable = Path(getattr(sys, "_base_executable", sys.executable)).resolve()
        executable_in_sandbox = Path("/runtime") / executable.relative_to(base)
        runner = _CHILD_RUNNER.format(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
        )

        command = [
            bwrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-net",
            "--clearenv",
        ]
        for system_path in ("/usr", "/lib", "/lib64", "/bin"):
            if Path(system_path).exists():
                command.extend(("--ro-bind", system_path, system_path))
        command.extend(
            (
                "--ro-bind",
                str(base),
                "/runtime",
                "--ro-bind",
                str(venv),
                "/venv",
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--dir",
                "/work",
                "--chdir",
                "/work",
                "--setenv",
                "HOME",
                "/tmp",
                "--setenv",
                "PATH",
                "/usr/bin:/bin",
                "--setenv",
                "PYTHONNOUSERSITE",
                "1",
                str(executable_in_sandbox),
                "-I",
                "-u",
                "-c",
                runner,
            )
        )
        return command

    def run(
        self,
        code: str,
        request_handler: Callable[[str, dict[str, Any]], Any],
    ) -> dict[str, Any]:
        started = time.monotonic()
        requests = 0
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as process_stderr:
            process = subprocess.Popen(
                self._sandbox_command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=process_stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
                preexec_fn=self._set_limits,
            )
            assert process.stdin is not None
            assert process.stdout is not None
            try:
                process.stdin.write(json.dumps({"code": code}) + "\n")
                process.stdin.flush()
                finished: dict[str, Any] | None = None
                while finished is None:
                    remaining = self.timeout_seconds - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("controller exceeded wall-clock limit")
                    ready, _, _ = select.select([process.stdout], [], [], remaining)
                    if not ready:
                        raise TimeoutError("controller exceeded wall-clock limit")
                    line = process.stdout.readline(self.max_line_bytes + 1)
                    if not line:
                        break
                    if len(line) > self.max_line_bytes:
                        raise RuntimeError("controller protocol message exceeded size limit")
                    message = json.loads(line)
                    if message.get("type") == "finished":
                        finished = message
                        break
                    if message.get("type") != "request":
                        raise RuntimeError("invalid controller protocol message")
                    requests += 1
                    if requests > self.max_requests:
                        raise RuntimeError("controller exceeded sensor/control request limit")
                    request_id = message.get("id")
                    try:
                        method = message["method"]
                        params = message.get("params", {})
                        if not isinstance(method, str) or not isinstance(params, dict):
                            raise TypeError("invalid sensor/control request")
                        result = request_handler(method, params)
                        response = {"id": request_id, "ok": True, "result": result}
                    except BaseException as exc:
                        response = {"id": request_id, "ok": False, "error": str(exc)}
                    process.stdin.write(json.dumps(response, separators=(",", ":")) + "\n")
                    process.stdin.flush()

                process.wait(timeout=5)
                process_stderr.seek(0)
                infrastructure_stderr = process_stderr.read()
                if finished is None:
                    return {
                        "ok": False,
                        "stdout": "",
                        "stderr": infrastructure_stderr
                        or f"restricted controller exited with status {process.returncode}",
                        "result": None,
                    }
                if infrastructure_stderr:
                    finished["stderr"] = finished.get("stderr", "") + infrastructure_stderr
                return {
                    "ok": bool(finished.get("ok")) and process.returncode == 0,
                    "stdout": str(finished.get("stdout", "")),
                    "stderr": str(finished.get("stderr", "")),
                    "result": finished.get("result"),
                }
            except BaseException as exc:
                process.kill()
                process.wait(timeout=5)
                process_stderr.seek(0)
                infrastructure_stderr = process_stderr.read()
                return {
                    "ok": False,
                    "stdout": "",
                    "stderr": f"{type(exc).__name__}: {exc}\n{infrastructure_stderr}",
                    "result": None,
                }
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=5)


__all__ = ["RestrictedControllerExecutor"]
