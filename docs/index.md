# fieldtrial

**Find out whether your robot policy actually got better.**

fieldtrial is an open-source Python framework for statistically rigorous, real-world
evaluation of robot policies. LeRobot trains the policy; fieldtrial tells you whether it
actually got better.

!!! warning "Alpha"
    The [statistics core](stats/index.md) and the [calculator commands](cli.md) work.
    Studies, the operator console and reports follow in 0.1.0. See the
    [roadmap](ROADMAP.md).

## What it does

- **Before an evaluation:** how many rollouts do you need? What is the smallest difference
  this evaluation can detect?
- **During:** a randomized, blinded schedule, plus a phone-friendly operator console that
  logs the outcome, progress stage and failure tags of each rollout.
- **After:** correct statistics (exact tests, paired designs, multiplicity control), honest
  wording, and a shareable report.

## Principles

- **Correctness first.** Every statistical function is tested against an independent
  reference implementation.
- **Pre-registration.** The design is locked and hashed before the first trial.
- **Honest language.** A report calls an arm better only when the pre-registered test
  rejects.
- **Local-first.** One folder per study and SQLite storage. Works offline on a factory LAN.
- **No telemetry**, ever.
