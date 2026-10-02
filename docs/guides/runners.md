# Real blinding with runners

With the `manual` runner, the operator loads each arm's checkpoint, so they can know which
arm is running. A switching runner starts the policy itself, and the operator only ever
sees blind codes:

| Runner | For | What it does |
|---|---|---|
| `command` | Any policy you start with a command (`lerobot-rollout`, your own script) | Runs a command template for each trial, filled with the arm's `policy` and `serving` values |
| `openpi_router` | openpi-style websocket policy servers | Sits between the robot's client and one policy server per arm, and forwards each trial's frames to that trial's arm |

Either way, every arm uses the same runner. The console and the REST API both drive it:
**Start** starts the arm, **Stop** stops it, and **Invalid trial** stops it too. If a
runner cannot start a trial, for example because the command does not exist, the trial is
marked invalid with the reason and rescheduled.

Check the setup before a session:

```console
$ fieldtrial check-runners my-study
  K7       ok  runs lerobot-rollout (6 arguments)
  Q2       ok  runs lerobot-rollout (6 arguments)
```

The check names arms by blind code only, so it does not unblind anyone.

## The command runner

```yaml
arms:
  - id: baseline
    runner: command
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 30}
  - id: q50
    runner: command
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference.type: rtc, inference.queue_threshold: 50}

runners:
  command:
    template: >-
      lerobot-rollout --policy.path={policy.path} --robot.type=so101_follower
      --inference.type={serving[inference.type]}
      --inference.queue_threshold={serving[inference.queue_threshold]}
    stop_signal: SIGINT        # like Ctrl-C; then SIGTERM, then SIGKILL
    grace_s: 10
```

The template is split into arguments like a shell would, but it never runs through a
shell. Placeholders are filled in each argument, so a path with spaces stays one argument:

| Placeholder | Value |
|---|---|
| `{policy.KEY}`, `{policy[KEY]}` | the arm's `policy` value |
| `{serving.KEY}`, `{serving[KEY]}` | the arm's `serving` value (keys may contain dots) |
| `{factors.NAME}` | the trial's condition, for example `{factors.slot}` |
| `{blind_code}`, `{trial_id}`, `{seq}`, `{condition}`, `{instruction}`, `{timeout_s}`, `{study}` | the trial and study |

The arm id is not available: anyone at the robot can see the command line. An unknown
placeholder is an error in `fieldtrial validate`, before any trial runs. `true` and
`false` are written in lower case, and `{{` and `}}` give literal braces.

The command also gets these environment variables: `FIELDTRIAL_STUDY`, `FIELDTRIAL_ARM`
(the blind code), `FIELDTRIAL_TRIAL_ID`, `FIELDTRIAL_SEQ` and `FIELDTRIAL_CONDITION`. Its
output goes to `logs/<trial id>.log` in the study folder. Other settings:
`cwd` (relative to the study folder), `env` (extra variables) and `success_exit_code`.

**Stopping LeRobot cleanly.** `lerobot-rollout` (LeRobot 0.6.1) treats SIGINT, SIGTERM,
SIGHUP and SIGQUIT as a request to shut down: it ends the rollout and runs its teardown,
which can return the arm to its starting position. Keep `grace_s` long enough for that.
The default of 10 seconds covers the 3-second return.

**Outcome from the exit code.** With `success_exit_code: 0`, a command that exits with 0
gets the success stage preselected in the label form. The operator still confirms the
label. Use this when your script judges success itself.

**Process handling.** Each command gets its own process group, so the stop signal reaches
everything it started. The report's runner table counts abnormal exits per arm: trials
whose command ended by itself with a non-zero code.

### Loading the model once

Loading a large model for every trial is slow. With `launch: per_arm`, the command keeps
running while consecutive trials use the same arm, which is common in
[crossover rounds](../stats/crossover.md). It is told about each trial on its standard
input:

```text
start {"trial_id": "…", "seq": 7, "condition": "slot=3", "factors": {"slot": 3}}
stop
```

When the next trial needs another arm, fieldtrial closes the command's standard input,
which means "shut down", waits `grace_s`, and then signals it. Off-the-shelf rollout
commands do not speak this protocol, so `per_arm` needs a small wrapper around your
policy:

```python
import json
import sys

policy = load_policy()  # your code: load once
for line in sys.stdin:
    if line.startswith("start "):
        trial = json.loads(line[len("start ") :])
        start_rollout(policy, trial)  # your code: run until told to stop
    elif line.strip() == "stop":
        stop_rollout()
```

Measure how long one load takes before choosing. With randomized blocks the arm changes
most trials, so `per_trial` is usually as fast.

## The openpi router

Install the extra: `pip install 'fieldtrial[openpi]'`.

```yaml
arms:
  - id: baseline
    runner: openpi_router
    policy: {url: ws://gpu-box:8000}
  - id: candidate
    runner: openpi_router
    policy: {url: ws://gpu-box:8001}

runners:
  openpi_router:
    listen: 0.0.0.0:9000       # where the robot's client connects
    metadata: identical        # or: first
```

1. Start one policy server per arm, as usual.
2. Run `fieldtrial serve my-study`. The router starts listening with the server.
3. Point the robot's openpi client at the router (`WebsocketClientPolicy(host="router-host",
   port=9000)`) and leave it there.
4. Run trials in the console. At each Start, the router sends the robot's requests to
   that trial's arm.

**How it works.** For each robot connection the router opens one connection per arm. It
passes the client's `Api-Key` header through and sends the robot the servers' metadata.
After that it forwards every frame unchanged, without decoding it. If the arms' servers
send different metadata, the robot could tell them apart, so the router refuses the
connection. With `metadata: first`, it forwards the first arm's metadata instead and
records the difference. A server error, which openpi sends as a text frame, reaches the
robot's client unchanged and is counted.

Before the first trial starts, requests get an error ("no arm selected yet"). Between
trials they go to the arm of the previous trial. The router records the round-trip time
of every request during a trial. The report's runner table shows requests, errors and
latency per arm. This is a descriptive check that the arms were served alike, not a test.

## What runners do not hide

Runners keep the arm out of the operator's view, but anyone who reads the study
database, `study.yaml` or the server's process list can still find it. Real blinding also
needs the person who prepares the study to be someone other than the operator who labels
the trials.
