---
name: f1tenth-evosearch
description: Run an auditable K=8 F1TENTH controller-improvement campaign with fixed seed partitions, monotonic incumbent retention, skill promotion, and one-time held-out evidence.
---

# F1TENTH Evolutionary Search

The campaign state on disk is authoritative. Do not infer progress from memory.
At each turn, read `state.json`, the latest `decision.json`, and evidence from
development runs only.

## Improvement Discipline

1. Candidate A is the unchanged incumbent.
2. Candidates B-H each test a materially different mechanism.
3. Prefer mechanisms that explain failures: path direction, lookahead choice,
   curvature-aware speed, steering damping, recovery, or LiDAR clearance.
4. Evaluate all candidates on exactly the same development seeds.
5. Let the harness rank, validate, and accept. Never replace files manually.
6. Treat rejected hypotheses as evidence. Do not repeat them without a new
   mechanistic reason.
7. Promote a skill only through an accepted candidate's `skill.md`.

## Candidate Quality

Every policy must be complete, execute from reset, terminate when `obs["done"]`
is true, and set `RESULT`. Avoid seed conditionals and memorized command
sequences. A useful hypothesis predicts both why it should improve and what
trace evidence would falsify it.

## Evidence

Use `summary.json` for aggregate comparison. Use per-seed `result.json` for
collision, progress, and termination. Use `trace.json` and keyframes to inspect
API calls and vehicle state. Held-out artifacts are unavailable until the
controller and experimental contract are frozen.
