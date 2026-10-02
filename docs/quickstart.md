# Quickstart

## Install

fieldtrial needs Python 3.11 or newer. The quickest way is [uv](https://docs.astral.sh/uv/):

```console
$ uvx fieldtrial --version          # run without installing
$ uv tool install fieldtrial        # or install the command
$ pip install fieldtrial            # or into the current environment
```

## 1. Try it with a simulated robot

```console
$ uvx fieldtrial demo
```

This creates a blinded study comparing two serving settings of a simulated policy, with
known true success rates of 76% and 90%. Half of it is already run. The console opens in
your browser:

1. Start a session (any operator and rig name; tick the rig checklist).
2. Run the remaining trials: **Space** starts a trial, **Space** stops it, and **Enter**
   confirms the outcome the simulated runner suggests.
3. When every trial is done, open the report, tick the box and **Unblind**.
4. Open **Full report with charts**. Do the intervals cover 76% and 90%? With 40 trials per
   arm, the difference is probably not resolved, and the report says how large a
   difference this study could have detected.

## 2. Ask the calculator

No study needed:

```console
$ uvx fieldtrial compare 74/80 91/120     # did 74/80 beat 91/120?
$ uvx fieldtrial power --p1 0.76 --p2 0.90 # how many rollouts to detect 76% → 90%?
$ uvx fieldtrial mde --p1 0.76 --n 40      # what can 40 rollouts per arm detect?
```

See [Command line](cli.md) for every command.

## 3. Run your own study

```console
$ fieldtrial init my-study                # writes my-study/study.yaml
$ $EDITOR my-study/study.yaml             # arms, rubric, conditions, seed
$ fieldtrial validate my-study
$ fieldtrial plan my-study --baseline 0.75
$ fieldtrial lock my-study                # freezes the design and randomizes the schedule
$ fieldtrial serve my-study --lan         # scan the QR code with a phone at the robot
```

Then run the trials in the console. When they are done:

```console
$ fieldtrial unblind my-study
$ fieldtrial report my-study              # my-study/reports/report.html
```

Next: the [concepts](concepts.md) behind a study, the full [study workflow](studies.md),
and how to [choose the number of rollouts](guides/how-many-rollouts.md).
