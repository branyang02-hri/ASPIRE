#!/usr/bin/env python3
"""Launch a fresh F1TENTH ASPIRE campaign through a non-interactive Codex agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


SIM_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_policy_errors(event_log: Path, campaign: Path) -> list[str]:
    """Detect obvious cross-lineage reads or direct held-out evaluator calls."""
    errors: list[str] = []
    marker = "outputs/f1tenth/aspire-campaigns/"
    if not event_log.is_file():
        return ["agent event log is missing"]
    for line_number, line in enumerate(event_log.read_text().splitlines(), start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            errors.append(f"invalid JSON event at line {line_number}")
            continue
        item = event.get("item", {})
        if event.get("type") != "item.started" or item.get("type") != "command_execution":
            continue
        command = item.get("command", "")
        for referenced_id in re.findall(
            rf"{re.escape(marker)}([^/\s'\";|&()]+)", command
        ):
            if referenced_id != campaign.name:
                errors.append(
                    f"agent command at line {line_number} references another campaign: "
                    f"{referenced_id}"
                )
        if "evaluate_controller.py" in command and re.search(
            r"(?:--seeds\s+[^\n]*\b30[1-5]\b|seed[_ =]30[1-5]\b)", command
        ):
            errors.append(
                f"agent command at line {line_number} directly evaluates a held-out seed"
            )
    return errors


def build_prompt(
    campaign: Path,
    initial_controller: Path,
    max_iterations: int,
    task: str,
) -> str:
    return f"""
Run a complete, fresh ASPIRE F1TENTH evolutionary-search campaign.

The user has explicitly authorized this complete launch, including simulator
evaluations and model usage. Perform any required preflight, then proceed
immediately without asking for confirmation or stopping at a plan.

Working root: {SIM_ROOT}
Campaign path: {campaign}
Initial controller: {initial_controller}
Maximum iterations: {max_iterations}
Task: {task}

Read these files in full before acting:
- .claude/f1tenth/CLAUDE.md
- .claude/f1tenth/api-reference.md
- .claude/f1tenth/evosearch/INSTRUCTIONS.md
- .claude/f1tenth/evosearch/SKILL.md
- .claude/f1tenth/evosearch/main-agent-prompt.md
- .claude/f1tenth/evosearch/subagent-prompt.md

Execute the campaign end to end using scripts/f1tenth/campaign.py. Do not edit
the harness, simulator, evaluator, configuration, map, seed partitions, or
initial controller. No reference or oracle controller is supplied; derive
improvements from the documented API, development evidence, and retained history.
Write candidate files only in the prepared campaign iteration directories.
Use the full fixed development and validation batches. Do not inspect or run
held-out seeds until the harness has frozen the final controller. Preserve
failed candidates and rejected decisions. After the stop condition, freeze,
finalize the one-time held-out evaluation with videos, run campaign verify,
and return the campaign report path and concise result.

Pass `--task {task}` to `campaign.py init`. Do not substitute another task or
configuration.
""".strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--initial-controller",
        type=Path,
        default=Path(".claude/f1tenth/evosearch/initial_controller.py"),
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--task",
        choices=("levine_privileged", "spielberg_privileged"),
        default="levine_privileged",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--reasoning-effort", default="high")
    args = parser.parse_args()

    codex = shutil.which("codex")
    if codex is None:
        raise SystemExit("codex CLI is not installed")
    campaign = args.campaign.resolve()
    if campaign.exists() and any(campaign.iterdir()):
        raise SystemExit(f"campaign path is not empty: {campaign}")
    campaign.mkdir(parents=True, exist_ok=True)
    initial_controller = args.initial_controller.resolve()
    if not initial_controller.is_file():
        raise SystemExit(f"initial controller does not exist: {initial_controller}")

    log_dir = SIM_ROOT / "outputs/f1tenth/agent-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = log_dir / f"{campaign.name}.jsonl"
    final_message = log_dir / f"{campaign.name}-final.txt"
    prompt = build_prompt(
        campaign, initial_controller, args.max_iterations, args.task
    )
    command = [
        codex,
        "exec",
        "-m",
        args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "--sandbox",
        "workspace-write",
        "--json",
        "--output-last-message",
        str(final_message),
        "-C",
        str(SIM_ROOT),
        "-",
    ]
    with event_log.open("w") as log:
        process = subprocess.run(
            command,
            input=prompt,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=SIM_ROOT,
            check=False,
        )
    errors = []
    manifest_path = campaign / "manifest.json"
    if process.returncode == 0 and not manifest_path.is_file():
        errors.append("agent returned success without a campaign manifest")
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("agent") != {
            "model": args.model,
            "reasoning_effort": args.reasoning_effort,
        }:
            errors.append("campaign model provenance does not match launcher assignment")
        if manifest.get("max_iterations") != args.max_iterations:
            errors.append("campaign iteration budget does not match launcher assignment")
        if manifest.get("task") != args.task:
            errors.append("campaign task does not match launcher assignment")
    errors.extend(trace_policy_errors(event_log, campaign))

    independent_verification = None
    if manifest_path.is_file():
        verify = subprocess.run(
            [
                sys.executable,
                str(SIM_ROOT / "scripts/f1tenth/campaign.py"),
                "verify",
                "--campaign",
                str(campaign),
            ],
            cwd=SIM_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if verify.returncode != 0:
            errors.append(f"independent campaign verification exited {verify.returncode}")
        else:
            try:
                independent_verification = json.loads(verify.stdout)
            except json.JSONDecodeError:
                errors.append("independent campaign verification returned invalid JSON")
            else:
                if independent_verification.get("status") != "complete":
                    errors.append("campaign did not reach complete status")
                if not independent_verification.get("valid"):
                    errors.append("independent campaign verification failed")
    codex_version = subprocess.run(
        [codex, "--version"], capture_output=True, text=True, check=False
    ).stdout.strip()
    attestation = {
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "max_iterations": args.max_iterations,
        "task": args.task,
        "codex_version": codex_version,
        "return_code": process.returncode,
        "errors": errors,
        "independent_verification": independent_verification,
        "event_log": str(event_log),
        "event_log_sha256": sha256_file(event_log),
        "final_message": str(final_message),
        "final_message_sha256": sha256_file(final_message),
    }
    (campaign / "agent-run.json").write_text(
        json.dumps(attestation, indent=2, sort_keys=True) + "\n"
    )
    final_return_code = process.returncode if process.returncode else (3 if errors else 0)
    print(f"Codex return code: {process.returncode}")
    if errors:
        print(f"Attestation errors: {errors}")
    print(f"Event log: {event_log}")
    print(f"Final message: {final_message}")
    raise SystemExit(final_return_code)


if __name__ == "__main__":
    main()
