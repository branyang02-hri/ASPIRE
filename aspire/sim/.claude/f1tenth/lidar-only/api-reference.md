# F1TENTH LiDAR-Only API

Controller programs execute in a separate restricted process. They receive no
simulator object, global observation, map, reference path, pose, progress, reward,
overhead rendering, raw observation, or evaluator internals.

`get_scan()` returns a dictionary with exactly these fields:

- `scan`: `ranges`, `angle_min`, `angle_increment`, `range_min`, and `range_max`
- `speed`: local `longitudinal` speed and `yaw_rate`
- `timestamp`: simulator time in seconds
- `done`: whether the episode has ended

`drive(steering_angle, speed, steps=1)` applies bounded steering and forward speed
for 1-50 control periods and returns the same restricted sensor packet.

`stop()` commands zero speed for one control period and returns the same packet.

Candidate code may import NumPy and standard-library computation modules. Its
filesystem contains the Python runtime and packages, an empty writable work area,
and no ASPIRE repository or network.
