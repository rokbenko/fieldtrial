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

## Recording datasets

If you record rollouts as LeRobot datasets, note the trial number from the console in the
episode's metadata or file name, so an outcome can be traced back to its episode.
Linking trials to dataset episodes automatically is planned for v0.2.

## Coming next

- **v0.2:** links between trials and LeRobot dataset episodes.
- **v0.3:** an in-process LeRobot runner built on `lerobot.rollout`.
