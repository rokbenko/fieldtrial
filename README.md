# fieldtrial

**Find out whether your robot policy actually got better.**

[![CI](https://github.com/rokbenko/fieldtrial/actions/workflows/ci.yml/badge.svg)](https://github.com/rokbenko/fieldtrial/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/rokbenko/fieldtrial/blob/main/LICENSE)

fieldtrial is an open-source Python framework for statistically rigorous, real-world
evaluation of robot policies. LeRobot trains the policy; fieldtrial tells you whether it
actually got better.

> **Status: alpha.** The statistics core and the calculator commands work. Studies, the
> operator console and reports follow in 0.1.0. See the [roadmap](https://github.com/rokbenko/fieldtrial/blob/main/docs/ROADMAP.md).

## Why

Real-world evaluation is slow, noisy and ad hoc. With 40 rollouts, the 95% confidence
interval for a success rate is about ±15 percentage points, so many apparent
improvements from a fine-tuning run are just noise. fieldtrial helps at every stage:

- **Before an evaluation:** how many rollouts do I need? What is the smallest
  difference this evaluation can detect?
- **During:** a randomized, blinded schedule, plus a phone-friendly operator console
  that logs the outcome, progress stage and failure tags of each rollout.
- **After:** correct statistics (exact tests, paired designs, multiplicity control),
  honest wording, and a shareable report.

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

$ uvx fieldtrial mde --p1 0.76 --n 40           # what 40 rollouts per arm can detect
+21.1 pp (to 0.9707) / −30.1 pp (to 0.4588)
```

The same functions are available from Python in `fieldtrial.stats`. Every one of them is
tested against an independent reference implementation.

`fieldtrial demo`, with a simulated study, the operator console and a report, arrives in
0.1.0.

## How it fits with LeRobot and openpi

fieldtrial complements LeRobot and openpi and never forks them. It is not a training
framework, a simulation benchmark, a robot driver, a labeling platform or a cloud
service. In manual mode, fieldtrial schedules and records the trials and you run the
robot however you like. Integrations with other stacks are adapters.

## Development

See [CONTRIBUTING.md](https://github.com/rokbenko/fieldtrial/blob/main/CONTRIBUTING.md).

## License

Apache-2.0. Copyright 2026 Rok Benko.

---

<sup>The name comes from agricultural field trials. At Rothamsted in the 1920s, R. A.
Fisher developed randomized block designs for field trials, and those are the designs
fieldtrial uses to compare robot policies.</sup>
