#!/usr/bin/env python3
"""Auditable ASPIRE campaign state machine for F1TENTH evolutionary search."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SIM_ROOT = Path(__file__).resolve().parents[2]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from scripts.f1tenth.evaluate_controller import (
    evaluate_controller,
    sha256_file,
    sha256_text,
    write_json_atomic,
)


DEFAULT_TASK = "levine_privileged"
TASKS = {
    "levine_privileged": {
        "config": Path("env_configs/f1tenth/levine_privileged.yaml"),
        "map_base": Path("assets/f1tenth/levine/levine"),
        "reference_path": Path("assets/f1tenth/levine/levine_reference_path.csv"),
    },
    "spielberg_privileged": {
        "config": Path("env_configs/f1tenth/spielberg_privileged.yaml"),
        "map_base": Path("assets/f1tenth/spielberg/Spielberg_map"),
        "reference_path": Path("assets/f1tenth/spielberg/spielberg_reference_path.csv"),
    },
}
DEFAULT_SKILLS = Path(".claude/f1tenth/skills")
DEVELOPMENT_SEEDS = list(range(101, 111))
VALIDATION_SEEDS = list(range(201, 206))
HELD_OUT_SEEDS = list(range(301, 306))
CANDIDATE_NAMES = [f"candidate_{letter}" for letter in "ABCDEFGH"]
SCHEMA_VERSION = 2
EPSILON = 1e-6
BANNED_CANDIDATE_TEXT = (
    "ORACLE_CODE",
    "oracle_code",
    "aspire.sim.cap.envs.tasks.f1tenth.",
)

Evaluator = Callable[..., dict[str, Any]]


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(SIM_ROOT))
    except ValueError:
        return str(path.resolve())


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=SIM_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def append_event(campaign: Path, event: dict[str, Any]) -> None:
    event = {"timestamp": now(), **event}
    destination = campaign / "events.jsonl"
    with destination.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def hash_tree(path: Path) -> dict[str, str]:
    return {
        str(item.relative_to(path)): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def combined_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def task_assets(task: str) -> tuple[Path, Path, Path]:
    if task not in TASKS:
        raise ValueError(f"unknown F1TENTH task: {task}")
    spec = TASKS[task]
    map_base = SIM_ROOT / spec["map_base"]
    return (
        map_base.with_suffix(".png"),
        map_base.with_suffix(".yaml"),
        SIM_ROOT / spec["reference_path"],
    )


def task_asset_hashes(task: str) -> dict[str, str]:
    png, yaml_path, reference_path = task_assets(task)
    return {
        "png_sha256": sha256_file(png),
        "yaml_sha256": sha256_file(yaml_path),
        "reference_path_sha256": sha256_file(reference_path),
    }


def load_manifest(campaign: Path) -> dict[str, Any]:
    path = campaign / "manifest.json"
    if not path.is_file():
        raise ValueError(f"campaign manifest does not exist: {path}")
    return json.loads(path.read_text())


def load_state(campaign: Path) -> dict[str, Any]:
    path = campaign / "state.json"
    if not path.is_file():
        raise ValueError(f"campaign state does not exist: {path}")
    return json.loads(path.read_text())


def save_state(campaign: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = now()
    write_json_atomic(campaign / "state.json", state)
    write_report(campaign, state)


def write_report(campaign: Path, state: dict[str, Any]) -> None:
    manifest = load_manifest(campaign)
    lines = [
        f"# F1TENTH Campaign: {manifest['campaign_id']}",
        "",
        f"- Status: {state['status']}",
        f"- Task: {manifest.get('task', DEFAULT_TASK)}",
        f"- Model: {manifest['agent']['model']}",
        f"- Reasoning effort: {manifest['agent']['reasoning_effort']}",
        f"- Iterations completed: {state['iterations_completed']}/{manifest['max_iterations']}",
        f"- Accepted candidates: {state['accepted_count']}",
        f"- Incumbent code: {state['incumbent']['code_path']}",
        f"- Development: {state['incumbent']['development']['successes']}/"
        f"{state['incumbent']['development']['trial_count']}, "
        f"mean reward {state['incumbent']['development']['mean_reward']:.6f}",
        f"- Validation: {state['incumbent']['validation']['successes']}/"
        f"{state['incumbent']['validation']['trial_count']}, "
        f"mean reward {state['incumbent']['validation']['mean_reward']:.6f}",
        "",
        "## Iterations",
        "",
    ]
    if not state["history"]:
        lines.append("No iterations completed.")
    for item in state["history"]:
        lines.extend(
            [
                f"### Iteration {item['iteration']:02d}",
                "",
                f"- Proposed winner: {item['proposed_winner']}",
                f"- Accepted: {str(item['accepted']).lower()}",
                f"- Reason: {item['reason']}",
                f"- Decision: {item['decision_path']}",
                "",
            ]
        )
    if state.get("held_out"):
        held_out = state["held_out"]
        lines.extend(
            [
                "## Held-Out Evaluation",
                "",
                f"- Successes: {held_out['successes']}/{held_out['trial_count']}",
                f"- Mean reward: {held_out['mean_reward']:.6f}",
                f"- Summary: {held_out['summary_path']}",
                "",
            ]
        )
    (campaign / "report.md").write_text("\n".join(lines) + "\n")


def _evaluate(
    evaluator: Evaluator,
    *,
    code: str,
    label: str,
    config: Path,
    seeds: list[int],
    output: Path,
    record_video: bool = False,
) -> dict[str, Any]:
    return evaluator(
        code=code,
        controller_label=label,
        config_path=config,
        seeds=seeds,
        output=output,
        record_video=record_video,
        resume=True,
    )


def initialize_campaign(
    *,
    campaign: Path,
    initial_controller: Path,
    model: str,
    reasoning_effort: str,
    max_iterations: int,
    task: str = DEFAULT_TASK,
    config: Path | None = None,
    evaluator: Evaluator = evaluate_controller,
) -> dict[str, Any]:
    """Create a fresh lineage and establish development/validation baselines."""
    campaign = campaign.resolve()
    initial_controller = initial_controller.resolve()
    if task not in TASKS:
        raise ValueError(f"unknown F1TENTH task: {task}")
    config = config or TASKS[task]["config"]
    config = config.resolve() if config.is_absolute() else (SIM_ROOT / config).resolve()
    if campaign.exists() and any(campaign.iterdir()):
        raise ValueError(f"campaign path is not empty: {campaign}")
    if not initial_controller.is_file():
        raise ValueError(f"initial controller does not exist: {initial_controller}")
    if not config.is_file():
        raise ValueError(f"config does not exist: {config}")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")

    campaign.mkdir(parents=True, exist_ok=True)
    incumbent_dir = campaign / "incumbent"
    incumbent_dir.mkdir()
    code = initial_controller.read_text()
    if any(token in code for token in BANNED_CANDIDATE_TEXT):
        raise ValueError("initial controller references the oracle implementation")
    compile(code, str(initial_controller), "exec")
    incumbent_path = incumbent_dir / "code.py"
    incumbent_path.write_text(code)
    versions = incumbent_dir / "versions"
    versions.mkdir()
    shutil.copy2(incumbent_path, versions / "initial.py")

    source_skills = (SIM_ROOT / DEFAULT_SKILLS).resolve()
    working_skills = campaign / "skill-library-working"
    shutil.copytree(source_skills, working_skills)
    (working_skills / "promotions").mkdir(exist_ok=True)

    prompt_paths = [
        SIM_ROOT / ".claude/f1tenth/evosearch/main-agent-prompt.md",
        SIM_ROOT / ".claude/f1tenth/evosearch/subagent-prompt.md",
        SIM_ROOT / ".claude/f1tenth/evosearch/INSTRUCTIONS.md",
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": campaign.name,
        "task": task,
        "created_at": now(),
        "git_commit": git_commit(),
        "config_path": rel(config),
        "config_sha256": sha256_file(config),
        "map_assets": task_asset_hashes(task),
        "seed_partitions": {
            "development": DEVELOPMENT_SEEDS,
            "validation": VALIDATION_SEEDS,
            "held_out": HELD_OUT_SEEDS,
        },
        "candidate_count": len(CANDIDATE_NAMES),
        "candidate_names": CANDIDATE_NAMES,
        "max_iterations": max_iterations,
        "epsilon": EPSILON,
        "initial_controller_path": rel(initial_controller),
        "initial_controller_sha256": sha256_text(code),
        "initial_skill_library_sha256": combined_hash(hash_tree(working_skills)),
        "prompt_sha256": {
            rel(path): sha256_file(path) for path in prompt_paths if path.is_file()
        },
        "agent": {"model": model, "reasoning_effort": reasoning_effort},
    }
    write_json_atomic(campaign / "manifest.json", manifest)

    development = _evaluate(
        evaluator,
        code=code,
        label="initial",
        config=config,
        seeds=DEVELOPMENT_SEEDS,
        output=campaign / "baseline/development",
    )
    validation = _evaluate(
        evaluator,
        code=code,
        label="initial",
        config=config,
        seeds=VALIDATION_SEEDS,
        output=campaign / "baseline/validation",
    )
    state = {
        "schema_version": SCHEMA_VERSION,
        "status": "active",
        "created_at": now(),
        "updated_at": now(),
        "iterations_completed": 0,
        "accepted_count": 0,
        "prepared_iteration": None,
        "incumbent": {
            "code_path": rel(incumbent_path),
            "code_sha256": sha256_text(code),
            "source": "initial",
            "development": development,
            "validation": validation,
        },
        "history": [],
        "frozen": None,
        "held_out_started": None,
        "held_out": None,
    }
    write_json_atomic(campaign / "state.json", state)
    append_event(
        campaign,
        {
            "type": "campaign_initialized",
            "initial_code_sha256": sha256_text(code),
            "development_score": development["objective_score"],
            "validation_score": validation["objective_score"],
        },
    )
    write_report(campaign, state)
    return state


def prepare_iteration(campaign: Path) -> Path:
    manifest = load_manifest(campaign)
    state = load_state(campaign)
    if state["status"] != "active":
        raise ValueError(f"campaign is not active: {state['status']}")
    iteration = state["iterations_completed"]
    if iteration >= manifest["max_iterations"]:
        raise ValueError("iteration budget is exhausted")
    if state["prepared_iteration"] is not None:
        existing = campaign / f"iterations/iter_{state['prepared_iteration']:02d}"
        raise ValueError(f"an iteration is already prepared: {existing}")

    iteration_dir = campaign / f"iterations/iter_{iteration:02d}"
    if iteration_dir.exists():
        raise ValueError(f"iteration directory already exists: {iteration_dir}")
    candidate_a = iteration_dir / "candidate_A"
    candidate_a.mkdir(parents=True)
    incumbent_path = SIM_ROOT / state["incumbent"]["code_path"]
    shutil.copy2(incumbent_path, candidate_a / "code.py")
    (candidate_a / "hypothesis.md").write_text(
        "# Incumbent control\n\nUnmodified incumbent used as the performance floor.\n"
    )
    write_json_atomic(
        iteration_dir / "iteration.json",
        {
            "iteration": iteration,
            "status": "prepared",
            "incumbent_sha256": state["incumbent"]["code_sha256"],
            "candidate_names": CANDIDATE_NAMES,
            "prepared_at": now(),
        },
    )
    state["prepared_iteration"] = iteration
    save_state(campaign, state)
    append_event(campaign, {"type": "iteration_prepared", "iteration": iteration})
    return iteration_dir


def validate_candidates(iteration_dir: Path, incumbent_hash: str) -> dict[str, dict[str, Any]]:
    discovered = sorted(
        path.name
        for path in iteration_dir.iterdir()
        if path.is_dir() and path.name.startswith("candidate_")
    )
    if discovered != CANDIDATE_NAMES:
        raise ValueError(f"expected candidates {CANDIDATE_NAMES}, found {discovered}")

    metadata: dict[str, dict[str, Any]] = {}
    code_hashes: set[str] = set()
    hypothesis_hashes: set[str] = set()
    for name in CANDIDATE_NAMES:
        directory = iteration_dir / name
        code_path = directory / "code.py"
        hypothesis_path = directory / "hypothesis.md"
        if not code_path.is_file() or not hypothesis_path.is_file():
            raise ValueError(f"{name} requires code.py and hypothesis.md")
        code = code_path.read_text()
        hypothesis = hypothesis_path.read_text().strip()
        if len(hypothesis) < 20:
            raise ValueError(f"{name} hypothesis is too short")
        if any(token in code for token in BANNED_CANDIDATE_TEXT):
            raise ValueError(f"{name} references the oracle implementation")
        compile(code, str(code_path), "exec")
        code_hash = sha256_text(code)
        hypothesis_hash = sha256_text(hypothesis.lower())
        if name == "candidate_A" and code_hash != incumbent_hash:
            raise ValueError("candidate_A must be an exact copy of the incumbent")
        if name != "candidate_A" and code_hash == incumbent_hash:
            raise ValueError(f"{name} does not differ from the incumbent")
        if code_hash in code_hashes:
            raise ValueError(f"{name} duplicates another candidate")
        if hypothesis_hash in hypothesis_hashes:
            raise ValueError(f"{name} duplicates another hypothesis")
        code_hashes.add(code_hash)
        hypothesis_hashes.add(hypothesis_hash)
        metadata[name] = {
            "code": code,
            "code_path": code_path,
            "code_sha256": code_hash,
            "hypothesis_path": hypothesis_path,
            "hypothesis_sha256": hypothesis_hash,
            "skill_path": directory / "skill.md",
        }
    return metadata


def _rank_key(item: tuple[str, dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
    name, candidate, summary = item
    code_lines = sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in candidate["code"].splitlines()
    )
    return (
        -int(summary["valid"]),
        -int(summary["successes"]),
        -float(summary["mean_reward"]),
        code_lines,
        name,
    )


def run_iteration(
    campaign: Path,
    *,
    evaluator: Evaluator = evaluate_controller,
) -> dict[str, Any]:
    manifest = load_manifest(campaign)
    state = load_state(campaign)
    iteration = state["prepared_iteration"]
    if state["status"] != "active" or iteration is None:
        raise ValueError("prepare an active iteration before running it")
    if iteration != state["iterations_completed"]:
        raise ValueError("prepared iteration does not match campaign state")
    iteration_dir = campaign / f"iterations/iter_{iteration:02d}"
    candidates = validate_candidates(iteration_dir, state["incumbent"]["code_sha256"])
    config = (SIM_ROOT / manifest["config_path"]).resolve()

    development_results: dict[str, dict[str, Any]] = {}
    for name in CANDIDATE_NAMES:
        development_results[name] = _evaluate(
            evaluator,
            code=candidates[name]["code"],
            label=f"iter_{iteration:02d}/{name}",
            config=config,
            seeds=manifest["seed_partitions"]["development"],
            output=iteration_dir / name / "development",
        )

    ranked = sorted(
        (
            (name, candidates[name], development_results[name])
            for name in CANDIDATE_NAMES
        ),
        key=_rank_key,
    )
    winner_name, winner, winner_development = ranked[0]
    incumbent_development = state["incumbent"]["development"]
    strict_improvement = (
        winner_development["valid"]
        and winner_development["objective_score"]
        > incumbent_development["objective_score"] + manifest["epsilon"]
    )

    winner_validation = None
    validation_non_regression = False
    accepted = False
    if strict_improvement:
        winner_validation = _evaluate(
            evaluator,
            code=winner["code"],
            label=f"iter_{iteration:02d}/{winner_name}",
            config=config,
            seeds=manifest["seed_partitions"]["validation"],
            output=iteration_dir / winner_name / "validation",
        )
        incumbent_validation = state["incumbent"]["validation"]
        validation_non_regression = (
            winner_validation["valid"]
            and winner_validation["successes"] >= incumbent_validation["successes"]
            and winner_validation["objective_score"] + manifest["epsilon"]
            >= incumbent_validation["objective_score"]
        )
        accepted = validation_non_regression

    if not strict_improvement:
        reason = "no strict development improvement"
    elif not validation_non_regression:
        reason = "validation regression or invalid validation result"
    else:
        reason = "strict development improvement with validation non-regression"

    leaderboard = [
        {
            "candidate": name,
            "code_sha256": candidate["code_sha256"],
            "successes": summary["successes"],
            "trial_count": summary["trial_count"],
            "mean_reward": summary["mean_reward"],
            "objective_score": summary["objective_score"],
            "valid": summary["valid"],
        }
        for name, candidate, summary in ranked
    ]
    decision = {
        "schema_version": SCHEMA_VERSION,
        "iteration": iteration,
        "incumbent_before_sha256": state["incumbent"]["code_sha256"],
        "incumbent_development_score": incumbent_development["objective_score"],
        "proposed_winner": winner_name,
        "winner_development": winner_development,
        "winner_validation": winner_validation,
        "strict_development_improvement": strict_improvement,
        "validation_non_regression": validation_non_regression,
        "accepted": accepted,
        "reason": reason,
        "leaderboard": leaderboard,
        "decided_at": now(),
    }
    decision_path = iteration_dir / "decision.json"
    write_json_atomic(decision_path, decision)

    promotion = {
        "iteration": iteration,
        "candidate": winner_name,
        "accepted": accepted,
        "source_code_sha256": winner["code_sha256"],
        "source_skill_path": rel(winner["skill_path"]),
        "promoted_path": None,
        "reason": reason,
        "timestamp": now(),
    }
    if accepted:
        version_path = campaign / f"incumbent/versions/iter_{iteration:02d}_{winner_name}.py"
        version_path.write_text(winner["code"])
        current_path = campaign / "incumbent/code.py"
        current_path.write_text(winner["code"])
        state["incumbent"] = {
            "code_path": rel(current_path),
            "code_sha256": winner["code_sha256"],
            "source": f"iter_{iteration:02d}/{winner_name}",
            "development": winner_development,
            "validation": winner_validation,
        }
        state["accepted_count"] += 1
        if winner["skill_path"].is_file():
            promoted_path = (
                campaign
                / f"skill-library-working/promotions/iter_{iteration:02d}_{winner_name}.md"
            )
            shutil.copy2(winner["skill_path"], promoted_path)
            promotion["promoted_path"] = rel(promoted_path)
        else:
            promotion["reason"] += "; no skill.md supplied, recorded as a no-op promotion"

    with (campaign / "skill-promotions.jsonl").open("a") as handle:
        handle.write(json.dumps(promotion, sort_keys=True, separators=(",", ":")) + "\n")

    history_item = {
        "iteration": iteration,
        "proposed_winner": winner_name,
        "accepted": accepted,
        "reason": reason,
        "decision_path": rel(decision_path),
        "incumbent_after_sha256": state["incumbent"]["code_sha256"],
    }
    state["history"].append(history_item)
    state["iterations_completed"] += 1
    state["prepared_iteration"] = None
    solved = (
        state["incumbent"]["development"]["successes"]
        == len(manifest["seed_partitions"]["development"])
        and state["incumbent"]["validation"]["successes"]
        == len(manifest["seed_partitions"]["validation"])
    )
    budget_exhausted = state["iterations_completed"] >= manifest["max_iterations"]
    if solved or budget_exhausted:
        state["status"] = "ready_to_freeze"
        state["stop_reason"] = (
            "verified development and validation completion"
            if solved
            else "iteration budget exhausted"
        )
    save_state(campaign, state)
    append_event(
        campaign,
        {
            "type": "iteration_decided",
            "iteration": iteration,
            "winner": winner_name,
            "accepted": accepted,
            "reason": reason,
        },
    )
    return decision


def freeze_campaign(campaign: Path, reason: str) -> dict[str, Any]:
    state = load_state(campaign)
    manifest = load_manifest(campaign)
    if state["status"] == "frozen":
        return state
    if state["status"] != "ready_to_freeze":
        raise ValueError(f"campaign is not ready to freeze: {state['status']}")
    frozen_dir = campaign / "frozen"
    frozen_dir.mkdir(exist_ok=False)
    incumbent = SIM_ROOT / state["incumbent"]["code_path"]
    frozen_code = frozen_dir / "code.py"
    shutil.copy2(incumbent, frozen_code)
    skills_hashes = hash_tree(campaign / "skill-library-working")
    contract = {
        "task": manifest.get("task", DEFAULT_TASK),
        "controller_sha256": sha256_file(frozen_code),
        "config_sha256": manifest["config_sha256"],
        "map_assets": manifest["map_assets"],
        "skill_library_sha256": combined_hash(skills_hashes),
        "prompt_sha256": manifest["prompt_sha256"],
        "seed_partitions": manifest["seed_partitions"],
        "model": manifest["agent"],
        "reason": reason,
        "frozen_at": now(),
    }
    contract["contract_sha256"] = combined_hash(contract)
    write_json_atomic(frozen_dir / "manifest.json", contract)
    state["status"] = "frozen"
    state["frozen"] = {
        "code_path": rel(frozen_code),
        "controller_sha256": contract["controller_sha256"],
        "contract_path": rel(frozen_dir / "manifest.json"),
        "contract_sha256": contract["contract_sha256"],
        "reason": reason,
    }
    save_state(campaign, state)
    append_event(campaign, {"type": "campaign_frozen", **state["frozen"]})
    return state


def finalize_campaign(
    campaign: Path,
    *,
    evaluator: Evaluator = evaluate_controller,
) -> dict[str, Any]:
    manifest = load_manifest(campaign)
    state = load_state(campaign)
    if state["status"] == "complete":
        raise ValueError("held-out evaluation has already completed")
    if state["status"] not in {"frozen", "evaluating_held_out"}:
        raise ValueError("freeze the campaign before held-out evaluation")
    frozen_code = SIM_ROOT / state["frozen"]["code_path"]
    if sha256_file(frozen_code) != state["frozen"]["controller_sha256"]:
        raise ValueError("frozen controller hash mismatch")
    contract = json.loads((campaign / "frozen/manifest.json").read_text())
    contract_hash = contract.pop("contract_sha256")
    if combined_hash(contract) != contract_hash:
        raise ValueError("frozen contract hash mismatch")

    held_out_dir = campaign / "held-out"
    if state["status"] == "frozen":
        if held_out_dir.exists():
            raise ValueError("unregistered held-out artifacts already exist")
        state["status"] = "evaluating_held_out"
        state["held_out_started"] = {
            "controller_sha256": state["frozen"]["controller_sha256"],
            "seeds": manifest["seed_partitions"]["held_out"],
            "started_at": now(),
        }
        save_state(campaign, state)
        append_event(campaign, {"type": "held_out_started", **state["held_out_started"]})
    else:
        registered = state.get("held_out_started") or {}
        if (
            registered.get("controller_sha256") != state["frozen"]["controller_sha256"]
            or registered.get("seeds") != manifest["seed_partitions"]["held_out"]
        ):
            raise ValueError("held-out resume identity mismatch")

    summary = _evaluate(
        evaluator,
        code=frozen_code.read_text(),
        label="frozen_final",
        config=(SIM_ROOT / manifest["config_path"]).resolve(),
        seeds=manifest["seed_partitions"]["held_out"],
        output=held_out_dir,
        record_video=True,
    )
    state["status"] = "complete"
    state["held_out"] = {
        **summary,
        "summary_path": rel(campaign / "held-out/summary.json"),
    }
    save_state(campaign, state)
    append_event(
        campaign,
        {
            "type": "held_out_completed",
            "successes": summary["successes"],
            "trial_count": summary["trial_count"],
            "mean_reward": summary["mean_reward"],
        },
    )
    return state


def verify_campaign(campaign: Path) -> dict[str, Any]:
    manifest = load_manifest(campaign)
    state = load_state(campaign)
    errors: list[str] = []
    config = SIM_ROOT / manifest["config_path"]
    if not config.is_file() or sha256_file(config) != manifest["config_sha256"]:
        errors.append("config hash mismatch")
    task = manifest.get("task", DEFAULT_TASK)
    if task not in TASKS:
        errors.append("unknown task")
        current_map_hashes = {}
    else:
        current_map_hashes = task_asset_hashes(task)
    if current_map_hashes != manifest["map_assets"]:
        errors.append("map asset hash mismatch")
    incumbent = SIM_ROOT / state["incumbent"]["code_path"]
    if not incumbent.is_file() or sha256_file(incumbent) != state["incumbent"]["code_sha256"]:
        errors.append("incumbent hash mismatch")
    if manifest["seed_partitions"] != {
        "development": DEVELOPMENT_SEEDS,
        "validation": VALIDATION_SEEDS,
        "held_out": HELD_OUT_SEEDS,
    }:
        errors.append("seed partition mismatch")
    if len(state["history"]) != state["iterations_completed"]:
        errors.append("iteration history length mismatch")
    for item in state["history"]:
        decision_path = SIM_ROOT / item["decision_path"]
        if not decision_path.is_file():
            errors.append(f"missing decision for iteration {item['iteration']}")
            continue
        decision = json.loads(decision_path.read_text())
        if len(decision.get("leaderboard", [])) != len(CANDIDATE_NAMES):
            errors.append(f"incomplete leaderboard for iteration {item['iteration']}")
        for candidate in decision.get("leaderboard", []):
            summary = (
                decision_path.parent
                / candidate["candidate"]
                / "development/summary.json"
            )
            if not summary.is_file():
                errors.append(
                    f"missing development summary for iteration {item['iteration']} "
                    f"{candidate['candidate']}"
                )
        if decision.get("accepted") and not (
            decision.get("strict_development_improvement")
            and decision.get("validation_non_regression")
        ):
            errors.append(f"invalid acceptance at iteration {item['iteration']}")
    held_out_dir = campaign / "held-out"
    if state["status"] not in {"complete", "evaluating_held_out"} and held_out_dir.exists():
        errors.append("held-out artifacts exist before campaign completion")
    if state["status"] == "evaluating_held_out" and not state.get("held_out_started"):
        errors.append("held-out evaluation lacks a registered start event")
    if state["status"] == "complete":
        held_out = state.get("held_out") or {}
        held_out_seeds = manifest["seed_partitions"]["held_out"]
        if held_out.get("identity", {}).get("seeds") != held_out_seeds:
            errors.append("held-out seed set mismatch")
        if held_out.get("trial_count") != len(held_out_seeds):
            errors.append("held-out trial count mismatch")
        if not all(
            trial.get("video") and Path(trial["video"]).is_file()
            for trial in held_out.get("trials", [])
        ):
            errors.append("held-out video evidence is incomplete")
        if held_out.get("identity", {}).get("code_sha256") != state["frozen"]["controller_sha256"]:
            errors.append("held-out controller does not match frozen controller")
    result = {
        "campaign": str(campaign),
        "status": state["status"],
        "valid": not errors,
        "errors": errors,
    }
    write_json_atomic(campaign / "verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init = subparsers.add_parser("init")
    init.add_argument("--campaign", type=Path, required=True)
    init.add_argument("--initial-controller", type=Path, required=True)
    init.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK)
    init.add_argument("--config", type=Path)
    init.add_argument("--model", default="gpt-5.5")
    init.add_argument("--reasoning-effort", default="high")
    init.add_argument("--max-iterations", type=int, default=5)

    for command in ("prepare", "run-iteration", "status", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--campaign", type=Path, required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--campaign", type=Path, required=True)
    freeze.add_argument("--reason", required=True)

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--campaign", type=Path, required=True)

    args = parser.parse_args()
    campaign = args.campaign.resolve()
    if args.command == "init":
        result = initialize_campaign(
            campaign=campaign,
            initial_controller=args.initial_controller,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_iterations=args.max_iterations,
            task=args.task,
            config=args.config,
        )
    elif args.command == "prepare":
        result = {"iteration_dir": str(prepare_iteration(campaign))}
    elif args.command == "run-iteration":
        result = run_iteration(campaign)
    elif args.command == "freeze":
        result = freeze_campaign(campaign, args.reason)
    elif args.command == "finalize":
        result = finalize_campaign(campaign)
    elif args.command == "verify":
        result = verify_campaign(campaign)
    else:
        result = load_state(campaign)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
