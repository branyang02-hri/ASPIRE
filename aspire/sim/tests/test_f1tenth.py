from __future__ import annotations

import hashlib

import numpy as np
import pytest

from aspire.sim.cap.envs.configs.instantiate import instantiate
from aspire.sim.cap.envs.configs.loader import DictLoader
from aspire.sim.cap.envs.simulators.f1tenth import (
    CANONICAL_LEVINE_PNG_SHA256,
    CANONICAL_LEVINE_YAML_SHA256,
    CANONICAL_SPIELBERG_PNG_SHA256,
    CANONICAL_SPIELBERG_YAML_SHA256,
    F1TENTH_EFFECTIVE_LEVINE_SHA256,
    F1TENTH_EFFECTIVE_SPIELBERG_SHA256,
    OrderedLapTracker,
    canonical_levine_assets,
    canonical_spielberg_assets,
)


def test_canonical_levine_assets_have_expected_hashes() -> None:
    map_base, reference_path = canonical_levine_assets()
    assert hashlib.sha256(map_base.with_suffix(".png").read_bytes()).hexdigest() == (
        CANONICAL_LEVINE_PNG_SHA256
    )
    assert hashlib.sha256(map_base.with_suffix(".yaml").read_bytes()).hexdigest() == (
        CANONICAL_LEVINE_YAML_SHA256
    )
    assert reference_path.is_file()


def test_canonical_spielberg_assets_have_expected_hashes() -> None:
    map_base, reference_path = canonical_spielberg_assets()
    assert hashlib.sha256(map_base.with_suffix(".png").read_bytes()).hexdigest() == (
        CANONICAL_SPIELBERG_PNG_SHA256
    )
    assert hashlib.sha256(map_base.with_suffix(".yaml").read_bytes()).hexdigest() == (
        CANONICAL_SPIELBERG_YAML_SHA256
    )
    assert hashlib.sha256(reference_path.read_bytes()).hexdigest() == (
        "4c647bac61d3c9ce69ffec26bcd570c07874f071ae40eea1e3d8e61517522c3a"
    )


@pytest.mark.parametrize("direction", [1, -1])
def test_ordered_lap_tracker_accepts_a_complete_lap_in_either_direction(direction: int) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    path = np.column_stack((5.0 * np.cos(angles), 5.0 * np.sin(angles)))
    tracker = OrderedLapTracker(
        path,
        start_radius=0.2,
        departure_radius=1.0,
        max_cross_track=0.2,
        max_step_progress=0.5,
    )

    state = None
    sequence = angles if direction == 1 else angles[::-1]
    for _ in range(2):
        for angle in sequence:
            state = tracker.update(5.0 * np.cos(angle), 5.0 * np.sin(angle))
            if state.completed:
                break
        if state.completed:
            break

    assert state is not None
    assert state.completed
    assert state.direction == direction
    assert state.ordered_gate_fraction == 1.0


@pytest.mark.integration
def test_f1tenth_native_wrapper_reset_step_and_render() -> None:
    pytest.importorskip("f110_gym")
    from aspire.sim.cap.envs.simulators.f1tenth import F1TenthLevineLowLevel

    env = F1TenthLevineLowLevel(max_steps=100, render_size=192)
    try:
        observation, info = env.reset(seed=101, options={"start_pose": [0.0, 0.0, 0.0]})
        assert info["seed"] == 101
        assert observation["scan"]["ranges"].shape == (1080,)
        effective_map = env._env.sim.agents[0].scan_simulator.map_img
        assert hashlib.sha256(effective_map.tobytes()).hexdigest() == (
            F1TENTH_EFFECTIVE_LEVINE_SHA256
        )
        observation, reward, terminated, truncated, _ = env.step([0.0, 1.0])
        assert observation["pose"].shape == (3,)
        assert 0.0 <= reward <= 1.0
        assert not terminated
        assert not truncated
        assert env.render().shape == (192, 192, 3)
    finally:
        env.close()


@pytest.mark.integration
def test_spielberg_native_wrapper_uses_original_runtime_map() -> None:
    pytest.importorskip("f110_gym")
    from aspire.sim.cap.envs.simulators.f1tenth import F1TenthSpielbergLowLevel

    env = F1TenthSpielbergLowLevel(max_steps=100, render_size=192)
    try:
        observation, info = env.reset(seed=101)
        assert info["seed"] == 101
        assert observation["scan"]["ranges"].shape == (1080,)
        assert observation["collision"] is False
        effective_map = env._env.sim.agents[0].scan_simulator.map_img
        assert hashlib.sha256(effective_map.tobytes()).hexdigest() == (
            F1TENTH_EFFECTIVE_SPIELBERG_SHA256
        )
        assert env.render().shape == (192, 192, 3)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    "config_path",
    [
        "env_configs/f1tenth/levine_privileged.yaml",
        "env_configs/f1tenth/spielberg_privileged.yaml",
    ],
)
def test_controller_executes_through_code_execution_env(config_path: str) -> None:
    pytest.importorskip("f110_gym")
    config = DictLoader.load(config_path)
    env = instantiate(config["env"])
    try:
        env.reset(seed=101)
        code = """
obs = get_observation()
obs = drive(0.0, 0.5)
RESULT = {"scan_count": len(obs["scan"]["ranges"]), "done": obs["done"]}
"""
        _, reward, terminated, truncated, info = env.step(code)
        assert info["sandbox_rc"] == 0, info["stderr"]
        assert reward < 1.0
        assert not terminated
        assert not truncated
        assert info["task_completed"] is False
        assert env._exec_globals["RESULT"]["scan_count"] == 1080
    finally:
        env.close()
