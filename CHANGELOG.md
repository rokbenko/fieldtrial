# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Before 1.0,
minor releases may contain breaking changes.

## [Unreleased]

### Added

- `fieldtrial.stats`: anytime-valid statistics.
  - `betting_cs` and `capital_process`: betting confidence sequences for bounded means
    (the hedged capital process of Waudby-Smith and Ramdas), valid at every sample size.
  - `paired_anytime_test`: a test of paired block differences that may stop after any
    block, with an anytime-valid p-value and confidence sequence.
  - `eliminate`: best-arm selection by successive elimination over blocks.
- Anytime-valid stopping (`analysis.stopping: {rule: anytime}`, 2 arms): the study is
  checked after every complete block and stops, cancelling the remaining trials, as soon as
  the betting test rejects. Reports show the confidence sequence block by block.
- Best-arm selection (`analysis.selection: {rule: elimination, delta: 0.05}`, several arms
  in randomized blocks): after every complete block, arms that another arm beats with
  confidence are dropped and their remaining trials cancelled; the study stops when one
  arm remains. Reports name a selected arm only when it is the single survivor.
- Automatic looks are recorded as `interim_look` (rule `anytime`) and `selection_look`
  events and name arms by blind code in the console; they run when a trial is completed in
  the console or the API, and in `fieldtrial simulate`, but not during CSV imports.
- `Results` gains optional `anytime` and `selection` blocks and the primary methods
  `anytime` and `selection` (schema version unchanged).
- Documentation: anytime-valid comparisons, best-arm selection, and an evaluation of STEP
  (Snyder et al., RSS 2025), which fieldtrial does not port because of its license.

## [0.2.0] - 2026-10-02

Second release: crossover rounds, checkpoint ladders and group-sequential stopping; real
blinding with switching runners; evaluation-camera capture, rig drift checks and links to
LeRobot datasets. Studies, design hashes and results files from 0.1 keep working.

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
- Real blinding with switching runners. The console and the REST API start and stop them
  with each trial. `fieldtrial check-runners` checks the setup and names arms by blind
  code only.
  - `command` runner: runs `runners.command.template` for each trial with the arm's
    `policy` and `serving` values. The template is split like a shell would but never
    runs in one, and placeholders are checked at validation.
    - The command gets its own process group and is stopped with SIGINT, then SIGTERM,
      then SIGKILL. Logs go to `logs/`, and the environment carries the blind code only.
    - `launch: per_arm` keeps one process per arm and sends it start and stop lines.
    - With `success_exit_code`, a successful exit preselects the success stage.
  - `openpi_router` runner: a websocket proxy in front of one openpi policy server per
    arm.
    - It refuses arms whose metadata differ, passes `Api-Key` headers through and forwards
      frames unchanged to the current trial's arm.
    - It records request latency per trial. Install with `fieldtrial[openpi]`.
  - A runner that cannot start a trial marks it invalid with the reason, so the slot is
    rescheduled.
  - Reports gain a descriptive runner table per arm: requests, errors, latency and
    abnormal exits.
- Rig drift checks. Compare a photo of the rig with a reference photo: shift by phase
  correlation, brightness change, and structural similarity (SSIM) after alignment.
  - Run them with `fieldtrial rig-check DIR PHOTO` (or `--camera`), or attach a photo
    when a console session starts.
  - A flagged check shows as a console warning and as a report deviation, and the report
    lists every check.
- Evaluation-camera capture (`fieldtrial[capture]`: OpenCV and PyAV). With
  `capture.camera` in `study.yaml`, every trial is recorded from Start to Stop and the
  clip is attached to the trial. The rig is checked from the camera at session start and
  every `capture.drift.every_trials` trials. A camera failure never blocks a trial.
- Links between trials and LeRobot v3.0 dataset episodes (`fieldtrial[lerobot]`:
  pyarrow).
  - `fieldtrial link-episodes DIR DATASET` matches trials to episodes in run order, or
    from a `trial,episode_index` mapping file, and shows the plan by blind code before
    writing.
  - Episodes with DAgger `intervention` flags get intervention counts, and reports gain a
    descriptive dataset-episodes table.
- A `capture:` section in `study.yaml`, always left out of the design hash.
- `fieldtrial.stats`:
  - `spending_boundaries`, `constant_boundaries`, `crossing_probabilities`,
    `sequential_test`, `repeated_interval`
  - `trend_test`, `stratified_trend_test`, `plateau`, `plateau_paired`
  - `crossover_test`
- `Results` gains optional `ladder`, `crossover` and `sequential` blocks, the lists
  `runner`, `rig_checks` and `episodes` (empty by default), and the primary methods
  `ladder`, `crossover` and `group_sequential` (schema version unchanged: additions only).

### Changed

- `limits.reset: carry_over` is now accepted together with `design.type: crossover_rounds`.
- Design hashes of existing studies are unchanged: new settings are left out of the hash
  while they are unused.
- `pillow` is declared as a dependency; it was already required by matplotlib.

## [0.2.0rc1] - 2026-10-02

Release candidate of 0.2.0, published to PyPI for testing. Its changes are listed under
0.2.0; the release is identical apart from this changelog.

## [0.2.0a2] - 2026-10-02

Second v0.2 pre-release: real blinding with the command runner and the openpi router. Its
changes are listed under 0.2.0.

## [0.2.0a1] - 2026-10-02

First v0.2 pre-release: crossover rounds, checkpoint ladders and group-sequential stopping.
Its changes are listed under 0.2.0.

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

[Unreleased]: https://github.com/rokbenko/fieldtrial/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0...v0.2.0
[0.2.0rc1]: https://github.com/rokbenko/fieldtrial/compare/v0.2.0a2...v0.2.0rc1
[0.2.0a2]: https://github.com/rokbenko/fieldtrial/compare/v0.2.0a1...v0.2.0a2
[0.2.0a1]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0...v0.2.0a1
[0.1.0]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0a1...v0.1.0
[0.1.0rc1]: https://github.com/rokbenko/fieldtrial/compare/v0.1.0a1...v0.1.0rc1
[0.1.0a1]: https://github.com/rokbenko/fieldtrial/releases/tag/v0.1.0a1
