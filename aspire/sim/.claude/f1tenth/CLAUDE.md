# F1TENTH Suite Guide

F1TENTH has separate privileged and LiDAR-only contracts. The privileged tasks
expose full simulator state for integration testing. The constrained Levine and
Spielberg tasks provide only local vehicle sensing and actuation through an
isolated process.

## Environment

- Python: `.venv-f1tenth/bin/python`
- Setup: `bash scripts/f1tenth/setup_f1tenth.sh`
- Tasks: `levine_privileged`, `spielberg_privileged`, `levine_lidar_only`, `spielberg_lidar_only`
- Constrained configs: `env_configs/f1tenth/{levine,spielberg}_lidar_only.yaml`
- Constrained API: `.claude/f1tenth/lidar-only/api-reference.md`
- Simulator: pinned `cap/third_party/f1tenth_gym` submodule
- Canonical assets: `assets/f1tenth/{levine,spielberg}/`

The suite uses the canonical PNG and YAML tracked in ASPIRE and checks each selected track's hashes at runtime.

## Seed Partitions

- Development: 101-110
- Validation: 201-205
- Held-out evaluation: 301-305

Never inspect or run held-out seeds while developing candidates. Evaluate them once after selecting the final policy.

## APIs

Read [api-reference.md](api-reference.md) for privileged tasks and
[lidar-only/api-reference.md](lidar-only/api-reference.md) for the constrained
task. The constrained controller subprocess cannot mount this repository or use
the network. The constrained authoring agent receives only a fresh campaign,
API documentation, and a fixed campaign-command broker.

## Experiments

- Baseline evaluation: [run-baseline.md](run-baseline.md)
- Evolutionary search: [evosearch/INSTRUCTIONS.md](evosearch/INSTRUCTIONS.md)

The evolutionary-search campaign is enforced by
`scripts/f1tenth/campaign.py`. The coding agent authors candidates, but it
does not select winners, replace the incumbent, promote skills, or access
held-out seeds directly. Use `scripts/f1tenth/run_codex_campaign.py` for a
fresh end-to-end external-agent launch.
