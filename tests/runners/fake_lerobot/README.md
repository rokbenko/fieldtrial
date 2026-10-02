A fake `lerobot` (and `draccus`) package for the lerobot runner's tests.

It mirrors only the LeRobot 0.6.1 signatures that `fieldtrial.runners.lerobot_inprocess`
uses (checked against the 0.6.1 source); behaviour is minimal and every call is recorded in
`lerobot.CALLS`. The tests put this folder first on `sys.path` and remove the modules again
afterwards. The runner was also run against the real LeRobot 0.6.1 with a simulated robot
and small ACT checkpoints (see docs/guides/lerobot.md).
