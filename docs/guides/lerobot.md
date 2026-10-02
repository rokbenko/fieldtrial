# Using fieldtrial with LeRobot

LeRobot trains and runs the policy; fieldtrial decides what to run, in which order, and
whether the result means anything. They meet in the arms of `study.yaml`.

## Real blinding: the command runner

The `command` runner starts `lerobot-rollout` for each trial with the arm's checkpoint
and serving settings, so the operator only sees blind codes. See
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

If each trial recorded one episode of a LeRobot v3.0 dataset, link them (needs the
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

## Coming next

- **v0.3:** an in-process LeRobot runner built on `lerobot.rollout`.
