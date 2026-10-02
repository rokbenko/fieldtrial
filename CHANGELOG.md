# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0,
minor releases may contain breaking changes.

## [Unreleased]

### Added

- Crossover rounds (`design.type: crossover_rounds`) for tasks whose scene carries over
  between trials. Each arm runs whole rounds in cycles of two, in randomized, balanced
  order. The analysis is a period-adjusted difference with an exact randomization test,
  and its interval inverts the same test. A new `crossover-rounds` template. The console
  shows the round and when to reset the scene.
- Checkpoint ladders (`analysis.ladder`): Mantel's test of a linear association between
  training step and success, and plateau detection by fixed-sequence non-inferiority
  against the final checkpoint. Without a comparison, the association test is the
  primary analysis. Reports gain a success-by-step chart and a plateau table.
- Group-sequential stopping (`analysis.stopping: {rule: group_sequential, looks: K}`):
  - Lan–DeMets error-spending boundaries (O'Brien–Fleming or Pocock type).
  - Interim looks via `fieldtrial interim`, the console and `POST /api/v1/studies/{study}/interim`.
    A look reveals only "continue" or "stop" and cancels the remaining trials on a stop.
  - A stage-wise p-value and a repeated confidence interval in the final analysis.
  - The simulator runs planned looks as they come due (`--no-interim` to skip them).
- `fieldtrial.stats`:
  - `spending_boundaries`, `constant_boundaries`, `crossing_probabilities`,
    `sequential_test`, `repeated_interval`
  - `trend_test`, `stratified_trend_test`, `plateau`, `plateau_paired`
  - `crossover_test`
- `Results` gains optional `ladder`, `crossover` and `sequential` blocks, and the primary
  methods `ladder`, `crossover` and `group_sequential` (schema version unchanged:
  additions only).

### Changed

- `limits.reset: carry_over` is now accepted together with `design.type: crossover_rounds`.
- Design hashes of existing studies are unchanged: new settings are left out of the hash
  while they are unused.

## [0.1.0] - 2026-10-02

First release. Studies: design, lock, fill and analyze end to end, run them from a
phone-friendly operator console or your own runtime, and share a self-contained report.

### Added

- `study.yaml` schema with line-numbered validation errors, and three templates (`basic`,
  `checkpoint-ladder`, `serving-sweep`).
- Randomized complete block schedules with Williams-balanced arm order and blind codes.
  The schedule is identical on every platform and numpy version for a given seed.
- Locking and amendments: the design, its hash and the schedule are stored in a SQLite
  database in the study folder; every change is one transaction plus one event-log row.
- Invalid trials are kept and rescheduled at the end of their block.
- A simulated runner and auto-operator, and CSV import (all or nothing, `--map`) and
  CSV/JSONL export.
- The analysis engine: the primary analysis follows the locked design (McNemar with a Tango
  interval, CMH, or Cochran's Q with Holm; a threshold test for single-arm studies), with
  an independent Boschloo/Newcombe sensitivity analysis, stage funnels, time to success,
  per-condition and per-session tables, drift and invalid-trial checks, deviations and
  provenance. The `Results` model is versioned, and its JSON schema is published.
- Report wording from one tested template per situation, and a Markdown report.
- New statistics: stage distribution and funnel, Brunner–Munzel on stages, success-time
  curve and bootstrap median time, CMH test, and homogeneity tests for drift.
- Commands `init`, `validate`, `plan`, `lock`, `amend`, `simulate`, `import`, `export`,
  `status`, `unblind`, `analyze` and `report`.
- Documentation: "Running a study" and "Analysis and reports".
- The operator console (`fieldtrial serve`): start a session with the rig checklist, run
  trials by blind code with a timer, label the furthest stage, termination and failure tags,
  undo within 10 seconds, mark invalid trials, correct labels with a logged reason, a live
  mirror screen, keyboard and foot-pedal keys, light and dark themes. `--lan` serves phones
  on the local network behind an access token with a QR code.
- `fieldtrial demo`: a half-run simulated study in the console.
- REST API v1 with an OpenAPI document, Server-Sent Events, idempotency keys and optimistic
  concurrency, and `fieldtrial.client`, a dependency-free Python client.
- Reports list label corrections made after unblinding as deviations.
- Documentation: "Operator console", "REST API and client" and a phone test checklist.
- A self-contained HTML report with seven charts (`fieldtrial report`, now the default
  format, and in the console): no scripts and nothing loaded from other hosts.
- An example re-analysis of Dream Machines' published pi0.5 fine-tuning results.
- Documentation: quickstart, concepts, guides (planning, comparing checkpoints, ladders,
  serving sweeps, LeRobot, openpi, custom runtimes), and screenshots.
- `CITATION.cff`.

## [0.1.0rc1] - 2026-10-02

Release candidate of 0.1.0, published to PyPI for testing. Its changes are listed under
0.1.0; the release is identical apart from this changelog.

## [0.1.0a1] - 2026-10-01

First alpha: the statistics core and the calculator commands.

### Added

- `fieldtrial.stats`, the statistics core (numpy and scipy only):
  - one-arm intervals: Wilson (default), Clopper–Pearson, Jeffreys, Agresti–Coull, and an
    exact binomial test against a threshold
  - two independent arms: Newcombe difference CI, Boschloo exact test (primary), Fisher's
    exact test and the conditional odds ratio
  - paired designs: exact McNemar, Tango score CI, Cochran's Q with pairwise McNemar
  - multiplicity adjustments: Holm, Bonferroni, Benjamini–Hochberg
  - planning: sample size, power and minimum detectable effect (pooled-z, Fleiss
    continuity-corrected, arcsine; unequal allocation), exact power for Boschloo and
    McNemar, and a seeded simulation cross-check
  - Bayesian summaries (descriptive only): P(p_B > p_A) and credible intervals
- Calculator commands `fieldtrial ci`, `compare`, `paired`, `power`, `mde` and `adjust`,
  each with `--json` output.
- Documentation: a statistics reference page per method, a command-line page and the API
  reference.
- Project skeleton: packaging, `fieldtrial --version`, CI and the documentation site.

[Unreleased]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0a1...v0.1.0
[0.1.0rc1]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0a1...v0.1.0rc1
[0.1.0a1]: https://github.com/rokbenko/fieldtrial/releases/tag/v0.1.0a1
