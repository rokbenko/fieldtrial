# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0,
minor releases may contain breaking changes.

## [Unreleased]

Studies: design, lock, fill and analyze end to end, and run them from a phone-friendly
operator console or your own runtime.

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

[Unreleased]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/rokbenko/fieldtrial/releases/tag/v0.1.0a1
