# Using fieldtrial with LeRobot

LeRobot trains and runs the policy; fieldtrial decides what to run, in which order, and
whether the result means anything. They meet in the arms of `study.yaml`.

## Real blinding: the in-process runner

The `lerobot` runner runs each arm's policy inside fieldtrial itself. The robot is
connected once, the operator presses Start and Stop in the console and sees only blind
codes, and every trial becomes one episode of a LeRobot dataset, linked to the trial
automatically. It needs Python 3.12 or later and LeRobot 0.6:

```console
$ pip install 'fieldtrial[lerobot-runner]'
```

```yaml
arms:
  - id: baseline
    runner: lerobot
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference: {type: rtc, queue_threshold: 30}}
  - id: q50
    runner: lerobot
    policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
    serving: {inference: {type: rtc, queue_threshold: 50}}

runners:
  lerobot:
    robot:                       # LeRobot's robot configuration, as for --robot.*
      type: so101_follower
      port: /dev/ttyACM0
      id: lab_arm
      cameras:
        top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}
    fps: 30                      # policy and dataset rate
    keep_loaded: all             # policies kept in memory (a number, or all)
```

| Setting | Meaning |
|---|---|
| `policy.path`, `policy.revision` | The checkpoint (folder or Hub repo id) and, optionally, its revision |
| `serving.inference` | LeRobot's inference engine: `{type: sync}` (default) or `{type: rtc, queue_threshold: ..., rtc: {execution_horizon: ...}}` |
| `serving.interpolation_multiplier` | Commands per policy step (the robot is driven `fps` × this many times a second; the dataset still records `fps` frames a second) |
| `robot` | LeRobot's robot configuration with its `type`; third-party robots registered by an installed plugin work too |
| `fps` | Policy and dataset rate (default 30) |
| `device` | `cuda`, `mps` or `cpu`; by default the first policy's device, else the best available. Every arm uses the same one |
| `keep_loaded` | How many checkpoints stay in memory (default 1). Arms that share a checkpoint share its weights |
| `dataset` | `repo_id` (default `local/<study>`), `root` (default `lerobot/<name>` in the study folder), `video`, `streaming_encoding` |
| `rename_map` | Camera renames, as LeRobot's `--rename_map` |
| `reset_to_initial_position`, `reset_s` | After each trial, move the robot back to the pose it had when connected (default on, over 2 s) |
| `return_to_initial_position` | Do the same before disconnecting when the server stops (default on) |

**What happens in a trial.** The first trial connects the robot and opens the dataset
(creating it, or continuing it if it exists). Preparing an arm loads its checkpoint if it
is not loaded yet, and builds LeRobot's inference engine with the arm's settings; the
console shows "loading arm K7" meanwhile. Start runs the policy and records a frame at
`fps`; Stop ends the episode, saves it, writes `fieldtrial_episodes.json` (episode index,
trial id, trial number) next to the dataset's `meta/` folder, and records a
`dataset_link` event, so there is nothing to link by hand. A time limit
(`limits.timeout_s`) stops the policy and asks the operator to stop the trial. If the
control loop fails (a motor error, the inference engine), the robot stops being driven
and the console asks to void the trial.

The dataset is finalized after every episode, so it is valid between trials and can be
scored or reviewed while the study runs ([reward models](reward-models.md)). The episode's
task is the study's `task.instruction`, the same for every arm: nothing in the dataset
tells the arms apart.

**What it measures.** Each trial's `runner_output` records the frames, the recording
rate, control ticks that ran late, whether a policy had to be loaded first and how long
preparing took, and the policy's size. The report's **Runner** table summarises them per
arm, so the cost of switching arms is visible.

**Checking before a session.** `fieldtrial check-runners my-study` checks that LeRobot is
installed in a supported version and that every arm's policy configuration can be read,
without loading weights or touching the robot.

**What is and is not tested.** The runner mirrors the set-up of LeRobot 0.6.1's
`lerobot-rollout` (robot, dataset features, policy, processors, inference engine) and uses
its rollout helpers for the control loop; it refuses other LeRobot versions. fieldtrial's
own tests run it against a fake LeRobot with the same function signatures. It was also run
against LeRobot 0.6.1 itself, with a simulated robot and small ACT checkpoints on a CPU,
through the server's REST API. It has not been run on a real robot, with a GPU, or with RTC
inference on a policy that supports it. PEFT adapters and `torch.compile` are not supported.

## Blinding with your own command: the command runner

The `command` runner starts `lerobot-rollout` (or any script) for each trial with the
arm's checkpoint and serving settings, so the operator only sees blind codes. See
[Real blinding with runners](runners.md#the-command-runner).

## Manual mode

The operator starts each rollout with LeRobot and records the outcome in the console.

1. **Describe each arm** with what LeRobot needs to run it. `policy` and `serving` are
   free-form and passed through unchanged; they also end up in the report's provenance.

    ```yaml
    arms:
      - id: baseline
        policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
        serving: {inference.type: rtc, inference.queue_threshold: 30}
      - id: q50
        policy: {path: outputs/pi05_21h/checkpoints/050000/pretrained_model}
        serving: {inference.type: rtc, inference.queue_threshold: 50}
    ```

2. **Keep one launch command per blind code** next to the robot. After unblinding nobody
   needs the mapping; before, only the person preparing the commands should know it. For
   example, with LeRobot's rollout command and its real-time chunking settings:

    ```console
    $ lerobot-rollout --policy.path=outputs/pi05_21h/checkpoints/050000/pretrained_model \
        --robot.type=... --inference.type=rtc --inference.queue_threshold=50
    ```

    Check `lerobot-rollout --help` for the options of your LeRobot version.

3. **Run the console** (`fieldtrial serve my-study --lan`). For each trial it shows the
   blind code and the starting condition; the operator launches that command, presses
   Start when the robot starts and Stop when it ends, and labels the outcome.

The operator still knows which command they launched, so this is only half-blind. Keep
the launch scripts named by blind code, not by arm.

## Linking trials to dataset episodes

With the `lerobot` runner, trials are linked as they run. If each trial recorded one
episode of a LeRobot v3.0 dataset some other way, link them (needs the
`lerobot` extra: `pip install 'fieldtrial[lerobot]'`):

```console
$ fieldtrial link-episodes my-study outputs/eval_dataset      # or a cached repo id: lab/cups-eval
Dataset outputs/eval_dataset (v3.0)
  trial    1 (N4, completed) -> episode 0 (412 frames)
  trial    2 (Y4, completed) -> episode 1 (388 frames, 23 intervention frames)
  ...
Link 40 trials to these episodes? [y/N]:
```

LeRobot episodes carry no wall-clock time, so trials are matched to episodes in run order:
the first unlinked trial to `--first-episode` (0 by default), and so on. If a rollout was
restarted or an episode discarded, give the pairs yourself with `--map links.csv`, a file
with the columns `trial,episode_index`. `trial` is the trial number shown in the console,
or the trial id. The plan shows blind codes only, and nothing is written until you
confirm. Each linking run is recorded in the event log.

When the dataset was recorded with the DAgger rollout strategy, frames where a human took
over are tagged `intervention=True`. fieldtrial counts them per trial, and the report's
**Dataset episodes** table lists, per arm, the linked trials, frames, trials with
interventions and intervention frames. The table is descriptive; no test is run on it.

A repo id is looked up where LeRobot caches datasets (`$HF_LEROBOT_HOME/<repo_id>`, by
default `~/.cache/huggingface/lerobot/<repo_id>`); fieldtrial never downloads anything.
