# F1TENTH Evolutionary Search

This is the canonical ASPIRE experiment protocol for improving a privileged
Levine controller. The coding agent proposes programs; the campaign harness
owns evaluation, selection, lineage, and held-out access.

## Fixed Contract

- Development seeds: 101-110.
- Validation seeds: 201-205.
- Held-out seeds: 301-305.
- Population: eight complete candidates per iteration.
- Candidate A: exact incumbent.
- Candidates B-H: seven distinct, falsifiable hypotheses.
- Default budget: five iterations.
- Acceptance: strict development objective improvement, valid execution, and
  validation non-regression.
- Completion: a frozen controller, one held-out evaluation, five videos, and a
  passing campaign verification.

The scalar objective is success rate plus 0.001 times mean shaped reward. This
makes completed laps dominant while retaining partial progress as a tie-breaker.
Simulator `task_completed` is authoritative; no model judges success.

## Required Reading

From `aspire/sim`, read:

1. `README.md`
2. `.claude/f1tenth/CLAUDE.md`
3. `.claude/f1tenth/api-reference.md`
4. this file and `SKILL.md`
5. `main-agent-prompt.md`
6. `subagent-prompt.md`
7. `clean-task-slate.md` before rerunning an experiment

## Start A Fresh Campaign

Choose a new, empty campaign path. Never reuse another campaign's candidates,
skills, state, or results.

```bash
CAMPAIGN=outputs/f1tenth/aspire-campaigns/<campaign-id>

PYGLET_HEADLESS=true .venv-f1tenth/bin/python \
  scripts/f1tenth/campaign.py init \
  --campaign "$CAMPAIGN" \
  --initial-controller .claude/f1tenth/evosearch/initial_controller.py \
  --model gpt-5.5 \
  --reasoning-effort high \
  --max-iterations 5
```

Initialization hashes the repository contract, copies a fresh skill library,
and evaluates the initial controller on development and validation. It does not
touch held-out seeds.

## Iteration Loop

Prepare the next iteration:

```bash
.venv-f1tenth/bin/python scripts/f1tenth/campaign.py prepare \
  --campaign "$CAMPAIGN"
```

The harness creates `iter_NN/candidate_A`. Fill candidates B-H using
`subagent-prompt.md`. Each candidate requires:

```text
candidate_X/
  code.py
  hypothesis.md
  skill.md
```

Then run the complete paired evaluation:

```bash
PYGLET_HEADLESS=true .venv-f1tenth/bin/python \
  scripts/f1tenth/campaign.py run-iteration \
  --campaign "$CAMPAIGN"
```

Read `decision.json`, traces, and per-seed results. If the campaign remains
`active`, prepare the next iteration and derive new hypotheses from the
recorded evidence. Never edit an evaluated candidate or decision.

The harness changes the incumbent only after a verified acceptance. Rejected
candidates remain on disk and the incumbent hash stays unchanged. Only an
accepted candidate's `skill.md` can enter the campaign skill library.

## Freeze And Held-Out Evaluation

The harness changes state to `ready_to_freeze` only after perfect development
and validation completion or exhaustion of the declared iteration budget.

```bash
.venv-f1tenth/bin/python scripts/f1tenth/campaign.py freeze \
  --campaign "$CAMPAIGN" \
  --reason "<verified completion or budget exhaustion>"

PYGLET_HEADLESS=true .venv-f1tenth/bin/python \
  scripts/f1tenth/campaign.py finalize \
  --campaign "$CAMPAIGN"

.venv-f1tenth/bin/python scripts/f1tenth/campaign.py verify \
  --campaign "$CAMPAIGN"
```

`finalize` is the only campaign command that can run held-out seeds. It rejects
a changed frozen controller or contract and refuses a second held-out run.

## Codex Launch

The external-agent launcher uses the Codex CLI's existing secure
authentication. It does not accept or persist an API key:

```bash
PYGLET_HEADLESS=true .venv-f1tenth/bin/python \
  scripts/f1tenth/run_codex_campaign.py \
  --campaign outputs/f1tenth/aspire-campaigns/<campaign-id> \
  --model gpt-5.5 \
  --reasoning-effort high \
  --max-iterations 5
```

## Output Contract

```text
<campaign>/
  manifest.json
  state.json
  events.jsonl
  report.md
  baseline/
  incumbent/code.py
  incumbent/versions/
  iterations/iter_NN/candidate_A..H/
  skill-library-working/
  skill-promotions.jsonl
  frozen/code.py
  frozen/manifest.json
  held-out/seed_301..305/
  verification.json
```

Every seed directory contains the exact controller, result, trace, final
overhead frame, any API keyframes produced by the policy, and, for held-out
trials, video evidence.
