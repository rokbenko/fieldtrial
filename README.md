# fieldtrial

**Find out whether your robot policy actually got better.**

[![CI](https://github.com/rokbenko/fieldtrial/actions/workflows/ci.yml/badge.svg)](https://github.com/rokbenko/fieldtrial/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/fieldtrial.svg)](https://pypi.org/project/fieldtrial/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/rokbenko/fieldtrial/blob/main/LICENSE)

fieldtrial is an open-source Python framework for statistically rigorous, real-world
evaluation of robot policies. LeRobot trains the policy; fieldtrial tells you whether it
actually got better.

## Why

Real-world evaluation is slow, noisy and ad hoc. With 40 rollouts, the 95% confidence
interval for a success rate is 12 to 15 percentage points wide on each side, so many apparent
improvements from a fine-tuning run are just noise. fieldtrial helps at every stage:

- **Before:** how many rollouts do I need? What is the smallest difference this
  evaluation can detect?
- **During:** a randomized, blinded schedule, and a phone-friendly operator console that
  records the outcome, the furthest stage reached and the failure mode of each rollout.
- **After:** the right statistics for the design (exact tests, paired analyses,
  multiplicity control), honest wording, and a self-contained report.

## 30-second demo

```console
$ uvx fieldtrial compare 74/80 91/120
Arm 1: 74/80 = 92.5% (95% CI 84.6%–96.5%)
Arm 2: 91/120 = 75.8% (95% CI 67.4%–82.6%)
Difference +16.7 pp, Newcombe 95% CI [+6.2 pp, +26.0 pp]
Boschloo p = 0.0020 (two-sided); rejects H0 at α = 0.05
Fisher p = 0.0022

$ uvx fieldtrial power --p1 0.76 --p2 0.90      # rollouts needed to detect 76% → 90%
112 per arm (pooled-z)

$ uvx fieldtrial demo                           # a simulated study in the console
```

`fieldtrial demo` opens a half-run, blinded study with a simulated robot. Run the rest
from the keyboard (Space, Space, Enter), unblind, and read the report.

<p>
  <img src="https://raw.githubusercontent.com/rokbenko/fieldtrial/main/docs/assets/console-running.png" alt="The operator console on a phone: a trial in progress with its blind code, timer and a large Stop button" width="260">
  <img src="https://raw.githubusercontent.com/rokbenko/fieldtrial/main/docs/assets/console-label.png" alt="Labelling a trial: furthest stage reached, why it ended, failure tags" width="260">
</p>
<p>
  <img src="https://raw.githubusercontent.com/rokbenko/fieldtrial/main/docs/assets/report.png" alt="The HTML report: summary, success rate per arm with confidence intervals, primary analysis and a forest plot" width="560">
</p>

<!-- A short GIF of a trial run in the console goes here. -->

## Quickstart

```console
$ uv tool install fieldtrial              # or: pip install fieldtrial
$ fieldtrial init my-study                # writes my-study/study.yaml
$ fieldtrial plan my-study --baseline 0.75
$ fieldtrial lock my-study                # freezes the design and randomizes the schedule
$ fieldtrial serve my-study --lan         # scan the QR code with a phone at the robot
$ fieldtrial unblind my-study
$ fieldtrial report my-study              # my-study/reports/report.html
```

A study is one folder: `study.yaml` (the design) and a SQLite database. Everything is
local and works offline; fieldtrial sends no telemetry.

- [Documentation](https://rokbenko.github.io/fieldtrial/): quickstart, concepts, guides
  and a statistics reference.
- [Re-analysis of Dream Machines' published pi0.5 results](https://github.com/rokbenko/fieldtrial/tree/main/examples/dream-machines-pi05):
  which of 30 published comparisons the data actually resolve.

## What you get

- **Design:**
  - randomized complete blocks with balanced arm order
  - crossover rounds for tasks whose scene carries over
  - checkpoint ladders
  - planned interim looks, or anytime-valid checks after every block, with early stopping
  - best-arm selection among serving configurations by successive elimination
  - blind codes, a design hash, locking and logged amendments
- **Real blinding:** a command runner that launches your rollout command for each arm,
  and an openpi router that sends each trial's requests to that arm's policy server.
- **Console:** sessions with a rig checklist, a timer, stage and failure-tag labels,
  10-second undo, invalid trials with automatic rescheduling, a live mirror screen,
  keyboard and foot-pedal keys, LAN access with a QR code.
- **Evidence:** an evaluation camera that records every trial, rig checks against a
  reference photo, and links from trials to LeRobot dataset episodes (with DAgger
  interventions).
- **Analysis:** the primary test follows from the locked design:
  - exact McNemar with a Tango interval, Cochran–Mantel–Haenszel, or Cochran's Q with Holm
  - group-sequential boundaries, and anytime-valid confidence sequences
  - the association of success with training step, and plateau detection
  - a period-adjusted crossover test

  Every analysis also gets an independent-samples sensitivity analysis, stage funnels,
  time to success, drift checks and a list of every deviation from the plan.
- **Reports:** self-contained HTML with charts, Markdown for pull requests, and a
  versioned JSON results model.
- **Integration:** a REST API with a dependency-free Python client for custom runtimes,
  CSV import and export, and `fieldtrial.stats` as a library.

Every statistical function is tested against an independent reference implementation.
Reports only describe a difference when the pre-registered test rejects; otherwise they
say what the study could have detected.

## How it fits with LeRobot and openpi

fieldtrial complements LeRobot and openpi and never forks them. It is not a training
framework, a simulation benchmark, a robot driver, a labeling platform or a cloud
service. In manual mode, fieldtrial schedules and records the trials and you run the
robot however you like. For real blinding, the `command` runner launches your rollout
command (for example `lerobot-rollout`) for each arm, and the `openpi_router` runner
routes an openpi client's traffic to each trial's policy server; see
[Real blinding with runners](https://rokbenko.github.io/fieldtrial/guides/runners/).

## How to cite

If fieldtrial helps your research, please cite it (see
[CITATION.cff](https://github.com/rokbenko/fieldtrial/blob/main/CITATION.cff)):

```bibtex
@software{benko_fieldtrial,
  author  = {Benko, Rok},
  title   = {fieldtrial: statistically rigorous real-world evaluation for robot policies},
  url     = {https://github.com/rokbenko/fieldtrial},
  license = {Apache-2.0}
}
```

For evaluation practice in general, see Kress-Gazit et al. (2024), *Robot Learning as an
Empirical Science: Best Practices for Policy Evaluation*, arXiv:2409.09491.

## Development

See [CONTRIBUTING.md](https://github.com/rokbenko/fieldtrial/blob/main/CONTRIBUTING.md).
