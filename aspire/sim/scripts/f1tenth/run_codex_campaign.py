#!/usr/bin/env python3
"""Launch a fresh F1TENTH ASPIRE campaign through a non-interactive Codex agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any


SIM_ROOT = Path(__file__).resolve().parents[2]
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from scripts.f1tenth.campaign import (
    TASKS,
    finalize_campaign,
    freeze_campaign,
    initialize_campaign,
    load_state,
    prepare_iteration,
    run_iteration,
    verify_campaign,
)

RESTRICTED_TASKS = {"levine_lidar_only", "spielberg_lidar_only"}


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


RESTRICTED_CLIENT = r'''#!/usr/bin/env python3
import argparse
import json
import socket
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument(
    "command",
    choices=("init", "prepare", "run-iteration", "status", "freeze", "finalize", "verify"),
)
parser.add_argument("--reason")
args = parser.parse_args()
request = {"command": args.command}
if args.reason is not None:
    request["reason"] = args.reason
with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
    client.connect("/broker/campaign.sock")
    client.sendall((json.dumps(request) + "\n").encode())
    response = b""
    while not response.endswith(b"\n"):
        chunk = client.recv(65536)
        if not chunk:
            break
        response += chunk
payload = json.loads(response)
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if payload.get("ok") else 1)
'''


def restricted_guide(task: str) -> str:
    track = task.removesuffix("_lidar_only").title()
    return f"""# Restricted {track} Evolutionary Search

Improve a reactive F1TENTH controller until it completes a {track} lap. Controller
programs receive only the LiDAR-only API in `api-reference.md`; the simulator and
scorer are deliberately outside this workspace.

The campaign harness, not the coding agent, owns evaluation, candidate ranking,
incumbent replacement, skill promotion, fixed seed partitions, and held-out access.
Use only these commands:

```bash
python3 campaign_cli.py init
python3 campaign_cli.py prepare
python3 campaign_cli.py run-iteration
python3 campaign_cli.py status
python3 campaign_cli.py freeze --reason "<recorded stop reason>"
python3 campaign_cli.py finalize
python3 campaign_cli.py verify
```

After `prepare`, author complete `code.py`, `hypothesis.md`, and `skill.md` files
for `campaign/iterations/iter_NN/candidate_B` through `candidate_H`. Candidate A
is the untouched incumbent. Each new candidate must test a distinct mechanism.
Read only this campaign's incumbent, accepted skills, hypotheses, decisions, and
evaluation summaries. Do not try to find maps, simulator source, evaluator source,
reference paths, old campaigns, or hidden controllers; they are not mounted.

Every controller is a top-level program that uses `get_scan()`, `drive()`, and
`stop()`, exits when `packet["done"]` is true, and assigns JSON-compatible `RESULT`.
The harness evaluates all eight candidates on development seeds, validates only a
strict winner, and retains it only on validation non-regression. Continue while
state is `active`. When state is `ready_to_freeze`, freeze using its stop reason,
finalize once, and verify.
"""


def restricted_prompt(task: str, max_iterations: int) -> str:
    track = task.removesuffix("_lidar_only").title()
    return f"""
Run one fresh ASPIRE {track} evolutionary-search campaign in this isolated workspace.
Read README.md and api-reference.md completely. Initialize through campaign_cli.py
with the fixed budget of {max_iterations} iterations, then coordinate the full
candidate/evaluation loop exactly as documented. Work only under /workspace/campaign.
Do not stop at a plan. The command broker is authoritative and is the only route to
the simulator and evaluator.
""".strip()


class RestrictedCampaignBroker:
    """Expose fixed campaign state transitions without exposing evaluator code."""

    def __init__(
        self,
        *,
        socket_path: Path,
        campaign: Path,
        initial_controller: Path,
        task: str,
        model: str,
        reasoning_effort: str,
        max_iterations: int,
    ) -> None:
        self.socket_path = socket_path
        self.campaign = campaign
        self.initial_controller = initial_controller
        if task not in RESTRICTED_TASKS:
            raise ValueError(f"task is not a restricted F1TENTH task: {task}")
        self.task = task
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_iterations = max_iterations
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._server: socket.socket | None = None

    @staticmethod
    def _state_view(state: dict[str, Any]) -> dict[str, Any]:
        incumbent = state.get("incumbent", {})

        def score_view(name: str) -> dict[str, Any] | None:
            score = incumbent.get(name)
            if not isinstance(score, dict):
                return None
            return {
                key: score.get(key)
                for key in (
                    "successes",
                    "trial_count",
                    "collisions",
                    "mean_reward",
                    "objective_score",
                    "valid",
                )
            }

        return {
            "status": state.get("status"),
            "prepared_iteration": state.get("prepared_iteration"),
            "iterations_completed": state.get("iterations_completed"),
            "accepted_count": state.get("accepted_count"),
            "stop_reason": state.get("stop_reason"),
            "incumbent": {
                "code_sha256": incumbent.get("code_sha256"),
                "source": incumbent.get("source"),
                "development": score_view("development"),
                "validation": score_view("validation"),
            },
        }

    def _dispatch(self, request: dict[str, Any]) -> Any:
        command = request.get("command")
        if command == "init":
            return self._state_view(
                initialize_campaign(
                    campaign=self.campaign,
                    initial_controller=self.initial_controller,
                    model=self.model,
                    reasoning_effort=self.reasoning_effort,
                    max_iterations=self.max_iterations,
                    task=self.task,
                )
            )
        if command == "prepare":
            iteration_dir = prepare_iteration(self.campaign)
            relative = iteration_dir.relative_to(self.campaign)
            return {"iteration_dir": f"/workspace/campaign/{relative}"}
        if command == "run-iteration":
            decision = run_iteration(self.campaign)
            return {
                "iteration": decision["iteration"],
                "proposed_winner": decision["proposed_winner"],
                "accepted": decision["accepted"],
                "reason": decision["reason"],
                "state": self._state_view(load_state(self.campaign)),
            }
        if command == "status":
            return self._state_view(load_state(self.campaign))
        if command == "freeze":
            reason = request.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("freeze requires a non-empty reason")
            return self._state_view(freeze_campaign(self.campaign, reason))
        if command == "finalize":
            return self._state_view(finalize_campaign(self.campaign))
        if command == "verify":
            return verify_campaign(self.campaign)
        raise ValueError(f"unsupported campaign command: {command}")

    def _serve(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            self._server = server
            server.bind(str(self.socket_path))
            server.listen(4)
            server.settimeout(0.25)
            while not self._stop.is_set():
                try:
                    connection, _ = server.accept()
                except TimeoutError:
                    continue
                with connection:
                    try:
                        handle = connection.makefile("r", encoding="utf-8")
                        line = handle.readline(1_000_001)
                        if not line or len(line) > 1_000_000:
                            raise ValueError("invalid broker request")
                        request = json.loads(line)
                        if not isinstance(request, dict):
                            raise TypeError("broker request must be an object")
                        payload = {"ok": True, "result": self._dispatch(request)}
                    except BaseException as exc:
                        payload = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    connection.sendall(
                        (json.dumps(payload, default=str) + "\n").encode()
                    )
            self._server = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        for _ in range(100):
            if self.socket_path.exists():
                return
            self._stop.wait(0.01)
        raise RuntimeError("restricted campaign broker did not start")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self.socket_path.unlink(missing_ok=True)


def create_restricted_workspace(campaign: Path, task: str) -> Path:
    workspace = SIM_ROOT / "outputs/f1tenth/agent-workspaces" / campaign.name
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError(f"restricted agent workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "campaign").mkdir()
    (workspace / "README.md").write_text(restricted_guide(task))
    shutil.copy2(
        SIM_ROOT / ".claude/f1tenth/lidar-only/api-reference.md",
        workspace / "api-reference.md",
    )
    client = workspace / "campaign_cli.py"
    client.write_text(RESTRICTED_CLIENT)
    client.chmod(0o755)
    return workspace


def restricted_codex_command(
    *,
    codex: Path,
    workspace: Path,
    campaign: Path,
    broker_dir: Path,
    auth_dir: Path,
    model: str,
    reasoning_effort: str,
) -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("bubblewrap is required for restricted agent execution")
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
    ]
    for system_path in ("/usr", "/lib", "/lib64", "/bin"):
        if Path(system_path).exists():
            command.extend(("--ro-bind", system_path, system_path))
    for system_file in (
        "/etc/hosts",
        "/etc/nsswitch.conf",
        "/etc/resolv.conf",
        "/etc/ssl",
        "/etc/ca-certificates",
    ):
        if Path(system_file).exists():
            command.extend(("--ro-bind", system_file, system_file))
    command.extend(
        (
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/opt",
            "--ro-bind",
            str(codex.resolve()),
            "/opt/codex",
            "--bind",
            str(workspace),
            "/workspace",
            "--bind",
            str(campaign),
            "/workspace/campaign",
            "--ro-bind",
            str(broker_dir),
            "/broker",
            "--bind",
            str(auth_dir),
            "/codex-home",
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "CODEX_HOME",
            "/codex-home",
            "/opt/codex",
            "exec",
            "-m",
            model,
            "-c",
            f'model_reasoning_effort="{reasoning_effort}"',
            "-c",
            "shell_environment_policy.inherit=none",
            "--dangerously-bypass-approvals-and-sandbox",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "--json",
            "--output-last-message",
            "/workspace/final.txt",
            "-C",
            "/workspace",
            "-",
        )
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument(
        "--initial-controller",
        type=Path,
    )
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument(
        "--task",
        choices=tuple(sorted(TASKS)),
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
    initial_controller_arg = args.initial_controller or TASKS[args.task]["initial_controller"]
    initial_controller = initial_controller_arg.resolve()
    if not initial_controller.is_file():
        raise SystemExit(f"initial controller does not exist: {initial_controller}")

    log_dir = SIM_ROOT / "outputs/f1tenth/agent-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    event_log = log_dir / f"{campaign.name}.jsonl"
    final_message = log_dir / f"{campaign.name}-final.txt"
    restricted = args.task in RESTRICTED_TASKS
    broker: RestrictedCampaignBroker | None = None
    broker_dir: Path | None = None
    auth_dir: Path | None = None
    auth_file: Path | None = None
    if restricted:
        workspace = create_restricted_workspace(campaign, args.task)
        broker_dir = Path(tempfile.mkdtemp(prefix="aspire-f1tenth-broker-"))
        auth_dir = Path(tempfile.mkdtemp(prefix="aspire-f1tenth-auth-"))
        auth_dir.chmod(0o700)
        auth_file = auth_dir / "auth.json"
        shutil.copy2(Path.home() / ".codex/auth.json", auth_file)
        auth_file.chmod(0o600)
        broker = RestrictedCampaignBroker(
            socket_path=broker_dir / "campaign.sock",
            campaign=campaign,
            initial_controller=initial_controller,
            task=args.task,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            max_iterations=args.max_iterations,
        )
        broker.start()
        prompt = restricted_prompt(args.task, args.max_iterations)
        command = restricted_codex_command(
            codex=Path(codex),
            workspace=workspace,
            campaign=campaign,
            broker_dir=broker_dir,
            auth_dir=auth_dir,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        process_env = {
            "PATH": "/usr/bin:/bin",
            "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
        }
        run_cwd = workspace
    else:
        workspace = None
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
        process_env = None
        run_cwd = SIM_ROOT
    try:
        auth_cleanup = None
        if auth_file is not None:
            # Codex reads authentication before its first model request. Remove the
            # pathname immediately afterward so agent-authored commands cannot read it.
            auth_cleanup = threading.Timer(1.0, auth_file.unlink, kwargs={"missing_ok": True})
            auth_cleanup.start()
        with event_log.open("w") as log:
            process = subprocess.run(
                command,
                input=prompt,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=run_cwd,
                env=process_env,
                check=False,
            )
    finally:
        if "auth_cleanup" in locals() and auth_cleanup is not None:
            auth_cleanup.cancel()
        if auth_file is not None:
            auth_file.unlink(missing_ok=True)
        if broker is not None:
            broker.close()
        if broker_dir is not None:
            shutil.rmtree(broker_dir, ignore_errors=True)
        if auth_dir is not None:
            shutil.rmtree(auth_dir, ignore_errors=True)
    if restricted and workspace is not None:
        restricted_final = workspace / "final.txt"
        if restricted_final.is_file():
            shutil.copy2(restricted_final, final_message)
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
        "restricted_agent_workspace": str(workspace) if workspace else None,
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
