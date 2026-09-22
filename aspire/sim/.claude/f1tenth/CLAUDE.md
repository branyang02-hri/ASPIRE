# F1TENTH Suite Guide

The initial F1TENTH integration is intentionally privileged. Generated programs may use LiDAR, global pose, velocity, collision state, lap progress, the Levine reference path, and overhead rendering. Results from this configuration measure ASPIRE integration and program improvement, not LiDAR-only autonomy or real-vehicle transfer.

## Environment

- Python: `.venv-f1tenth/bin/python`
- Setup: `bash scripts/f1tenth/setup_f1tenth.sh`
- Config: `env_configs/f1tenth/levine_privileged.yaml`
- Simulator: pinned `cap/third_party/f1tenth_gym` submodule
- Canonical assets: `assets/f1tenth/levine/`

Do not substitute the F1TENTH Gym bundled Levine PGM. The suite uses the canonical PNG and YAML tracked in ASPIRE and checks their hashes at runtime.

## Seed Partitions

- Development: 101-110
- Validation: 201-205
- Held-out evaluation: 301-305

Never inspect or run held-out seeds while developing candidates. Evaluate them once after selecting the final policy.

## APIs

Read [api-reference.md](api-reference.md). This suite currently permits privileged state. A future reduced interface should be added as a separate API/configuration rather than changing the meaning of this benchmark.

## Experiments

- Baseline evaluation: [run-baseline.md](run-baseline.md)
- Evolutionary search: [evosearch/INSTRUCTIONS.md](evosearch/INSTRUCTIONS.md)

The evolutionary-search campaign is enforced by
`scripts/f1tenth/campaign.py`. The coding agent authors candidates, but it
does not select winners, replace the incumbent, promote skills, or access
held-out seeds directly. Use `scripts/f1tenth/run_codex_campaign.py` for a
fresh end-to-end external-agent launch.
