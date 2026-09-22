# F1TENTH Baseline

Evaluate the deliberately weak initial controller across the development batch:

```bash
PYGLET_HEADLESS=true .venv-f1tenth/bin/python scripts/f1tenth/evaluate_controller.py \
  --controller .claude/f1tenth/evosearch/initial_controller.py \
  --seeds 101 102 103 104 105 106 107 108 109 110 \
  --output outputs/f1tenth/baseline
```

Evaluation directories have immutable code/config/seed identities and contain
the exact controller, one structured result and trace per seed, a final
overhead frame, and an aggregate summary. Use `--record-video` for final evidence;
recording every population member adds avoidable I/O and memory use.

No reference or oracle controller is shipped in the agent-readable workspace.
