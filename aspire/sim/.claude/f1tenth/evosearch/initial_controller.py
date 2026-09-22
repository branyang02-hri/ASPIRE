obs = get_observation()

for _ in range(1200):
    obs = drive(0.0, 1.0)
    if obs["done"]:
        break

RESULT = {
    "completed_lap": bool(obs["lap"]["completed"]),
    "termination_reason": obs["termination_reason"],
    "progress_fraction": float(obs["lap"]["progress_fraction"]),
}
