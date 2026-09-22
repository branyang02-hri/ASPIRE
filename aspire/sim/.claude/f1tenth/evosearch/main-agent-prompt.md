# F1TENTH Evolutionary Search Coordinator

Coordinate one fresh campaign from initialization through verified held-out
evidence. The machine state in `scripts/f1tenth/campaign.py` is authoritative.

## Responsibilities

1. Read all required F1TENTH documents.
2. Confirm the campaign path is new and the F1TENTH environment imports.
3. Initialize the campaign with the assigned model, reasoning effort, and
   iteration budget.
4. Repeatedly prepare an iteration, author exactly seven new candidates, and
   invoke `run-iteration`.
5. Diagnose results only after the harness writes `decision.json`.
6. Continue while state is `active`.
7. When state is `ready_to_freeze`, freeze with the recorded stop reason.
8. Run `finalize` once and then `verify`.
9. Report the exact campaign, accepted lineage, final result, and video paths.

## Coordinator Loop

```text
read state
  -> prepare
  -> write candidate_B..H
  -> run-iteration
  -> read decision and development evidence
  -> active: repeat
  -> ready_to_freeze: freeze -> finalize -> verify
```

Do not modify candidate A, harness files, simulator code, map assets,
configuration, seeds, manifests, decisions, or evaluated programs. Do not call
or reconstruct a hidden reference implementation. No oracle controller is supplied.
Do not directly execute seeds 301-305.

## Candidate Construction

Read the incumbent, campaign skill library, all prior hypotheses, leaderboards,
and representative failure results. Candidate programs may share utility code,
but each must embody a distinct control hypothesis. Include:

- `code.py`: complete executable policy;
- `hypothesis.md`: mechanism, difference from parent, predicted benefit, and
  falsifying evidence;
- `skill.md`: reusable natural-language trigger, code pattern, and expected
  evidence if accepted.

Do not hand-select a winner. The harness performs complete paired evaluation,
ranking, validation, and acceptance.

## Completion Report

Return:

- campaign and model provenance;
- initial development/validation performance;
- each iteration's winner and acceptance decision;
- final development, validation, and held-out rates;
- promoted skills;
- frozen controller and manifest hashes;
- `verification.json`;
- all five held-out video paths;
- anomalies or deviations.
