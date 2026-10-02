# Operator console

The console runs the schedule at the robot: it shows which arm to load, times each trial,
and records the outcome with a few taps or key presses. It is a web page served by your
laptop, made for a phone or tablet next to the rig, and it works offline.

## Try it

```console
$ fieldtrial demo
```

`demo` creates a study with a simulated robot whose arms have known true success rates
(76% and 90%), runs half of it, and opens the console. Run the rest there: the simulated
runner suggests each outcome, so Space, Space, Enter runs a trial. Then unblind on the
report page and check that the intervals cover the true rates.

## Serve a study

```console
$ fieldtrial serve my-study          # this laptop only: http://127.0.0.1:8765/
$ fieldtrial serve my-studies/       # a folder of studies: pick one in the list
$ fieldtrial serve my-study --lan    # phones and tablets on the same network
```

Lock the study first (`fieldtrial lock`). With `--lan`, the server prints a URL with an
access token and a QR code to scan. Anyone with that URL can record trials, so share it
only with your operators; a new token is made every time the server starts.

## Running trials

1. **Start a session** on the study page: operator, rig, and the `rig_checklist` from
   `study.yaml`. Every check must be ticked.
2. The console shows the trial number, the block, the condition, and the **blind code** of
   the arm to load, with the reset instructions and the timeout.
3. **Start** when the robot starts. A timer runs and turns amber at 80% of the timeout and
   red at the timeout.
4. **Stop** when the trial ends. The duration ends here, even if labelling takes longer.
5. Label it: the **furthest stage reached** (success means reaching the success stage),
   **why it ended**, any **failure tags**, and an optional note. **Confirm** moves to the
   next trial.
6. **Undo** is offered for 10 seconds after a confirm: the trial goes back to its label
   form, with the same stop time.

**Invalid trials** (a robot fault, a camera that fell) are marked with **Invalid trial…**
and a reason. They are kept and counted in the report, and the slot is rescheduled at the
end of its block.

**Corrections** after the fact go through **Trial history**: pick a completed trial, change
its label, and give your name and a reason. The event log keeps the old and new label.
Corrections made after unblinding are listed as deviations in every report.

**Evaluation camera.** With `capture.camera` in `study.yaml`, every trial is recorded from
Start to Stop and the clip is attached to the trial. If the study has a rig reference
photo, the session form offers a **Rig photo** field (phones open the camera). Without a
photo, the rig is checked from the camera. A flagged check shows as a warning above the
trial. See [Evaluation camera and rig checks](guides/capture.md).

**Switching runners** (`command`, `openpi_router`) start and stop the arm with **Start**
and **Stop**, and their state appears above the trial ("Runner: …"). If a runner cannot
start, the trial is marked invalid with the reason and rescheduled. With
`success_exit_code`, a successful command preselects the success stage; you still confirm
the label.

**Crossover rounds** show "round r of R" instead of the block. The first trial of a round
says to reset the whole scene and load the round's arm; the other trials say not to reset.

**Interim looks.** In a group-sequential study, a banner appears between trials when a
planned interim look is due. **Run interim look** shows only "continue" or "stop", so the
study stays blinded. After a stop, the remaining trials are cancelled and the console
points to the report.

## Keys and foot pedals

| Action | Default key |
|---|---|
| Start / Stop | Space |
| Stage reached | 0 = none, 1–9 = stage number |
| Confirm | Enter |
| Invalid | I |
| Undo | U |

Keys are ignored while you type in a text field. To change a key (for example for a USB
foot pedal, which sends a key press), open **Keyboard and foot pedal**, click the action's
box and press the new key. Bindings are saved on that device.

## Second screen

**Open a mirror screen** shows the same session in large type, read-only, for a monitor the
operator can see from the robot. It follows the console live, as does any other device
showing the session.

## Blinding

While a study is blinded (`blinding: operator`), the console, the history and the API show
blind codes only, and the report page offers nothing but the logged **Unblind** step.

!!! warning "Manual mode only half-blinds"
    With the `manual` runner, the operator loads the checkpoint and can know which arm is
    running. For real blinding, use a [switching runner](guides/runners.md).

## Safety and privacy

- On 127.0.0.1 the server only answers requests addressed to this machine, so web pages
  cannot reach it through DNS tricks.
- Every change carries a CSRF token, and every page has a strict Content-Security-Policy.
  Pages load nothing from other hosts.
- Double taps and retried requests are applied once (idempotency keys). Two devices
  labelling the same trial cannot overwrite each other: the second gets an error.
- fieldtrial sends no telemetry. FastAPI's optional OpenTelemetry hooks are switched off.

A [checklist for testing the console on a phone](guides/console-checklist.md) covers these
steps on real devices.
