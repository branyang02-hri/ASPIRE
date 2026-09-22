#!/usr/bin/env python3
"""Replay one frozen F1TENTH controller over an immutable seed batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from aspire.sim.cap.envs.configs.instantiate import instantiate
from aspire.sim.cap.envs.configs.loader import DictLoader
SCHEMA_VERSION = 2


def is_restricted_config(config: dict[str, Any]) -> bool:
    apis = config.get("env", {}).get("cfg", {}).get("apis", [])
    return any(str(name).startswith("F1TenthLidarApi") for name in apis)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    temporary.replace(path)


def _reset_trace_loggers(env: Any) -> None:
    for api in getattr(env, "_apis", {}).values():
        if hasattr(api, "get_trace_logger"):
            api.get_trace_logger().reset()


def _save_trace_loggers(env: Any, output: Path) -> None:
    for api in getattr(env, "_apis", {}).values():
        if hasattr(api, "get_trace_logger"):
            api.get_trace_logger().save(output)
            api.get_trace_logger().reset()


def run_trial(
    config: dict[str, Any],
    code: str,
    seed: int,
    output: Path,
    video: bool,
) -> dict[str, Any]:
    """Execute one replay and preserve its controller, result, trace, and video."""
    trial_dir = output / f"seed_{seed:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "code.py").write_text(code)
    restricted = is_restricted_config(config)

    env = instantiate(config["env"])
    try:
        _reset_trace_loggers(env)
        env.enable_video_capture(video, clear=True)
        _, reset_info = env.reset(seed=seed)
        initial_observation = env.low_level_env.get_observation()
        _, reward, terminated, truncated, info = env.step(code)
        observation = env.low_level_env.get_observation()
        result: dict[str, Any] = {
            "seed": seed,
            "reward": float(reward),
            "completed": bool(info.get("task_completed", False)),
            "collision": bool(observation["collision"]),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "sandbox_rc": int(info["sandbox_rc"]),
            "stdout": info["stdout"],
            "stderr": info["stderr"],
            "result": info.get("result", env._exec_globals.get("RESULT")),
        }
        if not restricted:
            result.update(
                {
                    "start_pose": reset_info.get(
                        "start_pose", initial_observation["pose"].tolist()
                    ),
                    "progress_fraction": float(
                        observation["lap"]["progress_fraction"]
                    ),
                    "ordered_gate_fraction": float(
                        observation["lap"]["ordered_gate_fraction"]
                    ),
                    "cross_track_error": float(
                        observation["lap"]["cross_track_error"]
                    ),
                    "termination_reason": str(observation["termination_reason"]),
                }
            )
        _save_trace_loggers(env, trial_dir)
        if not restricted:
            imageio.imwrite(trial_dir / "final_frame.png", env.render())
        if video:
            frames = env.get_video_frames(clear=True)
            if frames:
                video_path = trial_dir / "video.mp4"
                imageio.mimsave(video_path, frames, fps=5, macro_block_size=1)
                result["video"] = str(video_path)
        write_json_atomic(trial_dir / "result.json", result)
        return result
    finally:
        env.close()


def objective_score(success_rate: float, mean_reward: float) -> float:
    """Make lap completion dominant while retaining progress as a tie-breaker."""
    return float(success_rate + 0.001 * mean_reward)


def evaluate_controller(
    *,
    code: str,
    controller_label: str,
    config_path: Path,
    seeds: list[int],
    output: Path,
    record_video: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate a code snapshot, resuming only when its identity is unchanged."""
    seeds = sorted(set(int(seed) for seed in seeds))
    if not seeds:
        raise ValueError("at least one seed is required")
    config_path = config_path.resolve()
    identity = {
        "code_sha256": sha256_text(code),
        "config_sha256": sha256_file(config_path),
        "seeds": seeds,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        existing = json.loads(identity_path.read_text())
        if existing != identity:
            raise ValueError(f"evaluation identity mismatch: {identity_path}")
        if not resume:
            raise ValueError(f"evaluation already exists: {output}")
    else:
        write_json_atomic(identity_path, identity)

    config = DictLoader.load(str(config_path))
    results: list[dict[str, Any]] = []
    for seed in seeds:
        result_path = output / f"seed_{seed:03d}" / "result.json"
        if resume and result_path.is_file():
            result = json.loads(result_path.read_text())
        else:
            result = run_trial(config, code, seed, output, record_video)
        results.append(result)

    successes = sum(bool(result["completed"]) for result in results)
    mean_reward = sum(float(result["reward"]) for result in results) / len(results)
    success_rate = successes / len(results)
    valid = (
        len(results) == len(seeds)
        and all(int(result["sandbox_rc"]) == 0 for result in results)
        and all(math.isfinite(float(result["reward"])) for result in results)
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "identity": identity,
        "controller": controller_label,
        "trials": results,
        "trial_count": len(results),
        "successes": successes,
        "success_rate": success_rate,
        "collisions": sum(bool(result["collision"]) for result in results),
        "sandbox_failures": sum(int(result["sandbox_rc"]) != 0 for result in results),
        "mean_reward": mean_reward,
        "objective_score": objective_score(success_rate, mean_reward),
        "valid": valid,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("env_configs/f1tenth/levine_privileged.yaml"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    code = args.controller.read_text()
    summary = evaluate_controller(
        code=code,
        controller_label=str(args.controller),
        config_path=args.config,
        seeds=args.seeds,
        output=args.output,
        record_video=args.record_video,
        resume=not args.no_resume,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
