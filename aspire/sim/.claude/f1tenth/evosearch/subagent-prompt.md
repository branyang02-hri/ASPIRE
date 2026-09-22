# F1TENTH Candidate Author

Author one complete candidate under the iteration directory assigned by the
coordinator.

## Read

- `.claude/f1tenth/api-reference.md`
- `.claude/f1tenth/evosearch/SKILL.md`
- campaign `manifest.json` and `state.json`
- current incumbent
- campaign `skill-library-working/`
- previous `decision.json` files and development evidence

Do not read held-out outputs or seek a hidden reference implementation. No
oracle controller is supplied.

## Required Files

Write:

```text
candidate_X/code.py
candidate_X/hypothesis.md
candidate_X/skill.md
```

`code.py` must be a full top-level controller. It may import NumPy and standard
library modules, use only the documented F1TENTH API, stop after termination,
and assign a JSON-compatible `RESULT`.

`hypothesis.md` must state:

1. the parent or incumbent mechanism;
2. one falsifiable change;
3. why it should improve lap completion or progress;
4. the expected result/trace signature if it is wrong.

`skill.md` must state:

1. trigger condition;
2. reusable controller pattern or code excerpt;
3. why it should work;
4. evidence that would justify promotion.

Do not execute evaluation seeds yourself. The coordinator invokes the campaign
harness only after all eight candidates are complete.
