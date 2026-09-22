from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from scripts.f1tenth.campaign import (
    CANDIDATE_NAMES,
    finalize_campaign,
    freeze_campaign,
    initialize_campaign,
    load_state,
    prepare_iteration,
    run_iteration,
    verify_campaign,
)
from scripts.f1tenth.run_codex_campaign import trace_policy_errors
from scripts.f1tenth.evaluate_controller import objective_score, sha256_text, write_json_atomic


def fake_evaluator(
    *,
    code: str,
    controller_label: str,
    config_path: Path,
    seeds: list[int],
    output: Path,
    record_video: bool,
    resume: bool,
) -> dict[str, Any]:
    del config_path, resume
    dev_match = re.search(r"TEST_DEVELOPMENT=(\d+)", code)
    val_match = re.search(r"TEST_VALIDATION=(\d+)", code)
    held_match = re.search(r"TEST_HELD_OUT=(\d+)", code)
    if seeds[0] >= 300:
        desired = int(held_match.group(1) if held_match else val_match.group(1))
    elif seeds[0] >= 200:
        desired = int(val_match.group(1) if val_match else dev_match.group(1))
    else:
        desired = int(dev_match.group(1)) if dev_match else 0
    successes = min(desired, len(seeds))
    output.mkdir(parents=True, exist_ok=True)
    trials = []
    for index, seed in enumerate(seeds):
        completed = index < successes
        trial = {
            "seed": seed,
            "completed": completed,
            "collision": not completed,
            "reward": 1.0 if completed else 0.25,
            "sandbox_rc": 0,
        }
        if record_video:
            video = output / f"seed_{seed:03d}/video.mp4"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"video")
            trial["video"] = str(video)
        trials.append(trial)
    mean_reward = sum(item["reward"] for item in trials) / len(trials)
    summary = {
        "identity": {"code_sha256": sha256_text(code), "seeds": seeds},
        "controller": controller_label,
        "trials": trials,
        "trial_count": len(trials),
        "successes": successes,
        "success_rate": successes / len(trials),
        "collisions": len(trials) - successes,
        "sandbox_failures": 0,
        "mean_reward": mean_reward,
        "objective_score": objective_score(successes / len(trials), mean_reward),
        "valid": True,
    }
    write_json_atomic(output / "summary.json", summary)
    return summary


def write_candidate(directory: Path, name: str, development: int, validation: int) -> None:
    candidate = directory / name
    candidate.mkdir(parents=True, exist_ok=True)
    candidate.joinpath("code.py").write_text(
        f"# TEST_DEVELOPMENT={development}\n"
        f"# TEST_VALIDATION={validation}\n"
        f"# candidate={name}\n"
        "RESULT = {'candidate': '" + name + "'}\n"
    )
    candidate.joinpath("hypothesis.md").write_text(
        f"# {name}\n\nThis candidate tests distinct mechanism {name} with falsifiable evidence.\n"
    )
    candidate.joinpath("skill.md").write_text(
        f"# Skill from {name}\n\nTrigger, reusable pattern, rationale, and evidence.\n"
    )


def initialize(tmp_path: Path, *, max_iterations: int = 2) -> tuple[Path, str]:
    initial = tmp_path / "initial.py"
    initial.write_text(
        "# TEST_DEVELOPMENT=0\n# TEST_VALIDATION=0\n"
        "RESULT = {'candidate': 'initial'}\n"
    )
    campaign = tmp_path / "campaign"
    state = initialize_campaign(
        campaign=campaign,
        initial_controller=initial,
        config=Path("env_configs/f1tenth/levine_privileged.yaml"),
        model="gpt-5.5",
        reasoning_effort="high",
        max_iterations=max_iterations,
        evaluator=fake_evaluator,
    )
    return campaign, state["incumbent"]["code_sha256"]


def test_campaign_accepts_only_verified_improvement_and_finalizes_once(tmp_path: Path) -> None:
    campaign, initial_hash = initialize(tmp_path)
    iteration_dir = prepare_iteration(campaign)
    for index, name in enumerate(CANDIDATE_NAMES[1:], start=1):
        write_candidate(
            iteration_dir,
            name,
            development=10 if name == "candidate_B" else index,
            validation=5 if name == "candidate_B" else min(index, 4),
        )

    decision = run_iteration(campaign, evaluator=fake_evaluator)
    assert decision["accepted"] is True
    assert decision["proposed_winner"] == "candidate_B"
    state = load_state(campaign)
    assert state["status"] == "ready_to_freeze"
    assert state["incumbent"]["code_sha256"] != initial_hash
    assert state["accepted_count"] == 1
    promotions = [
        json.loads(line) for line in (campaign / "skill-promotions.jsonl").read_text().splitlines()
    ]
    assert promotions[0]["accepted"] is True
    assert promotions[0]["promoted_path"]

    freeze_campaign(campaign, "verified development and validation completion")
    finalize_campaign(campaign, evaluator=fake_evaluator)
    verification = verify_campaign(campaign)
    assert verification["valid"] is True
    assert load_state(campaign)["status"] == "complete"
    with pytest.raises(ValueError, match="already completed"):
        finalize_campaign(campaign, evaluator=fake_evaluator)


def test_validation_regression_keeps_incumbent(tmp_path: Path) -> None:
    campaign, initial_hash = initialize(tmp_path)
    state_path = campaign / "state.json"
    state = json.loads(state_path.read_text())
    state["incumbent"]["validation"]["successes"] = 5
    state["incumbent"]["validation"]["success_rate"] = 1.0
    state["incumbent"]["validation"]["mean_reward"] = 1.0
    state["incumbent"]["validation"]["objective_score"] = objective_score(1.0, 1.0)
    write_json_atomic(state_path, state)

    iteration_dir = prepare_iteration(campaign)
    for index, name in enumerate(CANDIDATE_NAMES[1:], start=1):
        write_candidate(
            iteration_dir,
            name,
            development=10 if name == "candidate_B" else index,
            validation=4,
        )
    decision = run_iteration(campaign, evaluator=fake_evaluator)
    assert decision["strict_development_improvement"] is True
    assert decision["validation_non_regression"] is False
    assert decision["accepted"] is False
    state = load_state(campaign)
    assert state["incumbent"]["code_sha256"] == initial_hash
    assert state["accepted_count"] == 0


def test_agent_trace_rejects_cross_lineage_and_direct_held_out_access(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "current-campaign"
    event_log = tmp_path / "events.jsonl"
    commands = [
        "cat outputs/f1tenth/aspire-campaigns/current-campaign/report.md",
        "cat outputs/f1tenth/aspire-campaigns/older-campaign/frozen/code.py",
        "python scripts/f1tenth/evaluate_controller.py --seeds 301 --controller code.py",
    ]
    event_log.write_text(
        "".join(
            json.dumps(
                {
                    "type": "item.started",
                    "item": {"type": "command_execution", "command": command},
                }
            )
            + "\n"
            for command in commands
        )
    )

    errors = trace_policy_errors(event_log, campaign)

    assert any("another campaign" in error for error in errors)
    assert any("held-out seed" in error for error in errors)
