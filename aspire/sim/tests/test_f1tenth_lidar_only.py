from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from aspire.sim.cap.envs.configs.instantiate import instantiate
from aspire.sim.cap.envs.configs.loader import DictLoader
from aspire.sim.cap.envs.tasks.f1tenth.restricted_executor import (
    RestrictedControllerExecutor,
)
from aspire.sim.cap.envs.simulators.f1tenth import (
    F1TENTH_EFFECTIVE_LEVINE_SHA256,
    F1TENTH_EFFECTIVE_SPIELBERG_SHA256,
)
from aspire.sim.cap.integrations.f1tenth.lidar_only import F1TenthLidarApi
from scripts.f1tenth.campaign import validate_controller_source
from scripts.f1tenth.evaluate_controller import evaluate_controller
from scripts.f1tenth.run_codex_campaign import (
    RestrictedCampaignBroker,
    restricted_codex_command,
    restricted_guide,
)


PUBLIC_PACKET_KEYS = {"scan", "speed", "timestamp", "done"}
PUBLIC_SCAN_KEYS = {
    "ranges",
    "angle_min",
    "angle_increment",
    "range_min",
    "range_max",
}


class FakeLowLevel:
    def __init__(self) -> None:
        self._raw_obs = {
            "scans": np.array([[1.0, 2.0, 3.0]]),
            "linear_vels_x": np.array([1.25]),
            "ang_vels_z": np.array([-0.2]),
        }
        self._sim_time = 2.5
        self._terminated = False
        self._truncated = False

    def step(self, action: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        del action
        self._sim_time += 0.01
        return {}, 0.0, self._terminated, self._truncated, {}


def test_lidar_api_has_an_exact_non_privileged_contract() -> None:
    api = F1TenthLidarApi(FakeLowLevel())  # type: ignore[arg-type]
    assert set(api.functions()) == {"get_scan", "drive", "stop"}
    packet = api.get_scan()
    assert set(packet) == PUBLIC_PACKET_KEYS
    assert set(packet["scan"]) == PUBLIC_SCAN_KEYS
    assert set(packet["speed"]) == {"longitudinal", "yaw_rate"}
    assert packet["scan"]["ranges"] == [1.0, 2.0, 3.0]
    serialized = json.dumps(packet)
    for forbidden in (
        "pose",
        "progress",
        "reward",
        "reference",
        "raw_observation",
        "collision",
    ):
        assert forbidden not in serialized


def test_restricted_executor_exposes_only_three_functions_and_result() -> None:
    packet = {
        "scan": {
            "ranges": [1.0],
            "angle_min": 0.0,
            "angle_increment": 1.0,
            "range_min": 0.0,
            "range_max": 30.0,
        },
        "speed": {"longitudinal": 0.0, "yaw_rate": 0.0},
        "timestamp": 0.0,
        "done": False,
    }

    def handler(method: str, params: dict[str, Any]) -> dict[str, Any]:
        assert method in {"get_scan", "drive", "stop"}
        assert isinstance(params, dict)
        return packet

    result = RestrictedControllerExecutor(timeout_seconds=10).run(
        """
visible = sorted(name for name in globals() if not name.startswith('__'))
RESULT = {"visible": visible, "packet": get_scan()}
""",
        handler,
    )
    assert result["ok"], result["stderr"]
    assert result["result"]["visible"] == ["RESULT", "drive", "get_scan", "stop"]
    assert set(result["result"]["packet"]) == PUBLIC_PACKET_KEYS


@pytest.mark.parametrize(
    "attempt",
    [
        "env",
        "APIS",
        "get_observation()",
        "get_reference_path()",
        "render_overhead()",
    ],
)
def test_privileged_names_are_absent_from_controller_process(attempt: str) -> None:
    result = RestrictedControllerExecutor(timeout_seconds=10).run(
        f"RESULT = {attempt}\n", lambda method, params: None
    )
    assert not result["ok"]
    assert "NameError" in result["stderr"]


def test_repository_map_and_python_modules_are_not_mounted() -> None:
    repository_map = Path(
        "/home/brandonyang/Documents/ChatGPT/AV Agent/ASPIRE/aspire/sim/"
        "assets/f1tenth/levine/levine.png"
    )
    code = f"""
import importlib.util
import os
RESULT = {{
    "map_exists": os.path.exists({str(repository_map)!r}),
    "aspire_module": importlib.util.find_spec("aspire") is not None,
    "cwd": os.getcwd(),
}}
"""
    result = RestrictedControllerExecutor(timeout_seconds=10).run(
        code, lambda method, params: None
    )
    assert result["ok"], result["stderr"]
    assert result["result"] == {
        "map_exists": False,
        "aspire_module": False,
        "cwd": "/work",
    }


def test_rpc_introspection_cannot_call_a_privileged_method() -> None:
    calls: list[str] = []

    def handler(method: str, params: dict[str, Any]) -> None:
        del params
        calls.append(method)
        if method not in {"get_scan", "drive", "stop"}:
            raise PermissionError(f"API method is not available: {method}")

    result = RestrictedControllerExecutor(timeout_seconds=10).run(
        """
try:
    get_scan.__globals__["_request"]("get_observation")
except RuntimeError as exc:
    RESULT = {"denied": "not available" in str(exc)}
""",
        handler,
    )
    assert result["ok"], result["stderr"]
    assert result["result"] == {"denied": True}
    assert calls == ["get_observation"]


@pytest.mark.parametrize("task", ["levine_lidar_only", "spielberg_lidar_only"])
def test_campaign_rejects_privileged_controller_symbols(task: str) -> None:
    for source in (
        "obs = get_observation()",
        "path = get_reference_path()",
        "frame = render_overhead()",
        "value = env.get_observation()",
        "value = APIS['anything']",
    ):
        with pytest.raises(ValueError, match="unavailable"):
            validate_controller_source(
                source, task=task, label="candidate"
            )


@pytest.mark.parametrize(
    ("task", "track"),
    [("levine_lidar_only", "Levine"), ("spielberg_lidar_only", "Spielberg")],
)
def test_restricted_guide_names_selected_track(task: str, track: str) -> None:
    guide = restricted_guide(task)
    assert f"Restricted {track} Evolutionary Search" in guide
    assert f"completes a {track} lap" in guide


def test_restricted_agent_mount_excludes_repository_and_other_outputs(
    tmp_path: Path,
) -> None:
    codex = shutil.which("codex")
    if codex is None or shutil.which("bwrap") is None:
        pytest.skip("Codex and bubblewrap are required")
    workspace = tmp_path / "workspace"
    campaign = tmp_path / "campaign"
    workspace.mkdir()
    campaign.mkdir()
    broker_dir = tmp_path / "broker"
    broker_dir.mkdir()
    auth_dir = tmp_path / "auth"
    auth_dir.mkdir()
    (workspace / "campaign").mkdir()
    (workspace / "README.md").write_text("restricted")
    command = restricted_codex_command(
        codex=Path(codex),
        workspace=workspace,
        campaign=campaign,
        broker_dir=broker_dir,
        auth_dir=auth_dir,
        model="gpt-5.5",
        reasoning_effort="high",
    )
    executable_index = len(command) - 1 - command[::-1].index("/opt/codex")
    probe = command[:executable_index] + [
        "/bin/sh",
        "-c",
        """
test -f /workspace/README.md &&
test -d /workspace/campaign &&
test ! -e '/home/brandonyang/Documents/ChatGPT/AV Agent/ASPIRE/aspire/sim/assets/f1tenth/levine/levine.png' &&
test ! -e '/home/brandonyang/Documents/ChatGPT/AV Agent/ASPIRE/aspire/sim/assets/f1tenth/spielberg/Spielberg_map.png' &&
test ! -e '/home/brandonyang/Documents/ChatGPT/AV Agent/ASPIRE/aspire/sim/scripts/f1tenth/evaluate_controller.py' &&
test ! -e '/home/brandonyang/Documents/ChatGPT/AV Agent/ASPIRE/aspire/sim/outputs/f1tenth/aspire-campaigns'
""",
    ]
    completed = subprocess.run(probe, capture_output=True, text=True, check=False)
    assert completed.returncode == 0, completed.stderr


def test_restricted_broker_state_view_omits_trial_details() -> None:
    state = {
        "status": "active",
        "prepared_iteration": 0,
        "iterations_completed": 0,
        "accepted_count": 0,
        "incumbent": {
            "code_sha256": "abc",
            "source": "initial",
            "development": {
                "successes": 0,
                "trial_count": 10,
                "collisions": 4,
                "mean_reward": 0.1,
                "objective_score": 0.0001,
                "valid": True,
                "trials": [{"privileged_detail": "must not cross broker"}],
            },
            "validation": None,
        },
    }
    view = RestrictedCampaignBroker._state_view(state)
    assert "trials" not in view["incumbent"]["development"]
    assert view["incumbent"]["development"]["collisions"] == 4


@pytest.mark.integration
@pytest.mark.parametrize(
    ("config_path", "track", "runtime_map_sha256"),
    [
        (
            "env_configs/f1tenth/levine_lidar_only.yaml",
            "levine",
            F1TENTH_EFFECTIVE_LEVINE_SHA256,
        ),
        (
            "env_configs/f1tenth/spielberg_lidar_only.yaml",
            "spielberg",
            F1TENTH_EFFECTIVE_SPIELBERG_SHA256,
        ),
    ],
)
def test_real_lidar_only_task_isolated_end_to_end(
    config_path: str,
    track: str,
    runtime_map_sha256: str,
) -> None:
    pytest.importorskip("f110_gym")
    config = DictLoader.load(config_path)
    env = instantiate(config["env"])
    try:
        observation, info = env.reset(seed=101)
        assert set(observation) == PUBLIC_PACKET_KEYS
        assert "start_pose" not in info
        assert track.title() in info["task_prompt"]
        assert env.low_level_env.track == track
        effective_map = env.low_level_env._env.sim.agents[0].scan_simulator.map_img
        assert hashlib.sha256(effective_map.tobytes()).hexdigest() == runtime_map_sha256
        code = """
packet = get_scan()
packet = drive(0.0, 0.5, steps=2)
RESULT = {"keys": sorted(packet), "count": len(packet["scan"]["ranges"])}
"""
        observation, _, _, _, step_info = env.step(code)
        assert step_info["sandbox_rc"] == 0, step_info["stderr"]
        assert step_info["result"] == {
            "keys": ["done", "scan", "speed", "timestamp"],
            "count": 1080,
        }
        assert set(observation) == PUBLIC_PACKET_KEYS
        assert env.low_level_env.get_observation()["pose"].shape == (3,)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_path",
    [
        "env_configs/f1tenth/levine_lidar_only.yaml",
        "env_configs/f1tenth/spielberg_lidar_only.yaml",
    ],
)
def test_restricted_evaluator_does_not_persist_privileged_trial_data(
    tmp_path: Path,
    config_path: str,
) -> None:
    pytest.importorskip("f110_gym")
    code = """
packet = get_scan()
while not packet["done"]:
    packet = drive(0.0, 1.0, steps=10)
RESULT = {"done": packet["done"]}
"""
    output = tmp_path / "evaluation"
    summary = evaluate_controller(
        code=code,
        controller_label="adversarial-artifact-check",
        config_path=Path(config_path),
        seeds=[101],
        output=output,
        record_video=False,
        resume=False,
    )
    trial = summary["trials"][0]
    assert not {
        "start_pose",
        "progress_fraction",
        "ordered_gate_fraction",
        "cross_track_error",
        "termination_reason",
    } & set(trial)
    assert not (output / "seed_101/final_frame.png").exists()
