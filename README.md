# fieldtrial

**Find out whether your robot policy actually got better.**

[![CI](https://github.com/rokbenko/fieldtrial/actions/workflows/ci.yml/badge.svg)](https://github.com/rokbenko/fieldtrial/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

fieldtrial is an open-source Python framework for statistically rigorous, real-world
evaluation of robot policies. LeRobot trains the policy; fieldtrial tells you whether it
actually got better.

> **Status: pre-alpha.** The project is being bootstrapped and nothing is usable yet.
> The statistics calculators arrive in 0.1.0a1. Studies, the operator console and
> reports follow in 0.1.0. See the [roadmap](docs/ROADMAP.md).

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

Coming with 0.1.0a1:

```bash
uvx fieldtrial compare 74/80 91/120
```

## How it fits with LeRobot and openpi

fieldtrial complements LeRobot and openpi and never forks them. It is not a training
framework, a simulation benchmark, a robot driver, a labeling platform or a cloud
service. In manual mode, fieldtrial schedules and records the trials and you run the
robot however you like. Integrations with other stacks are adapters.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Copyright 2026 Rok Benko.

---

<sup>The name comes from agricultural field trials. At Rothamsted in the 1920s, R. A.
Fisher developed randomized block designs for field trials, and those are the designs
fieldtrial uses to compare robot policies.</sup>
