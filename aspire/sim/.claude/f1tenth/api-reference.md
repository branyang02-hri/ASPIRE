# F1TENTH Privileged API

`get_observation()` returns LiDAR ranges and geometry, global pose, velocity, collision status, ordered lap progress, timing, previous command, termination state, and the raw F1TENTH Gym observation.

`get_reference_path()` returns the active track centerline as an `(N, 2)` NumPy array in world coordinates.

`drive(steering_angle, speed, steps=1)` applies bounded steering and forward-speed commands and returns the next full observation. Steering is clipped to `[-0.4189, 0.4189]` radians and speed to `[0, 8]` m/s.

`stop()` commands zero speed for one control period.

`render_overhead()` returns the current privileged overhead RGB image.

The generated namespace also contains `env`, following ASPIRE's existing `CodeExecutionEnvBase` convention.
