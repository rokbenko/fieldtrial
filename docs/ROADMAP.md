# Roadmap

The plan, its milestones and the decision log are in
[the project plan](PLAN.md). This page tracks progress and anything that is out of scope
for the current milestone.

## v0.1

| Milestone | Scope | Status |
|---|---|---|
| M0 | Bootstrap: packaging, CI, docs skeleton, `fieldtrial --version` | done |
| M1 | Statistics core and calculator CLI (release 0.1.0a1) | released 2026-10-02 |
| M2 | Study design, storage and analysis (no UI) | released in 0.1.0 |
| M3 | Operator console, REST API and Python client | released in 0.1.0 |
| M4 | Reports, docs and the Dream Machines re-analysis (release 0.1.0) | released 2026-10-02 |

## v0.2

| Milestone | Scope | Status |
|---|---|---|
| M5 | Crossover rounds, checkpoint ladders (step association, plateau), group-sequential stopping | released in 0.2.0 |
| M6 | Command-template runner and openpi router: real blinding | released in 0.2.0 |
| M7 | Evaluation-camera capture, rig drift check, LeRobot dataset links (release 0.2.0) | released 2026-10-02 |

Limits of M5, to revisit later:

- Crossover designs compare exactly 2 arms; Williams-type crossovers for more arms are
  not supported yet.
- Group-sequential stopping supports 2-arm randomized block designs with one replicate
  (the McNemar score statistic). CMH and multi-arm sequential designs, futility
  boundaries and stopping-adjusted point estimates are not supported yet.
- The command runner does not stop a trial when its command ends by itself; the operator
  presses Stop (the console shows "command ended"). An option to stop automatically could
  follow.
- The openpi router uses one connection per arm for each robot connection; servers that
  allow only one client at a time need one router per robot.
- Camera capture records one camera per study; several cameras, and audio, could follow.
- Dataset links match trials to episodes in run order or from a mapping file. A
  `fieldtrial` rollout strategy that writes the trial id into each episode would make the
  match exact (see the LeRobot runner notes).
- The ladder's plateau compares each checkpoint with the final one; a margin relative to
  the best checkpoint would need a different procedure.

### LeRobot runner spike (for v0.3)

These notes come from reading the LeRobot 0.6.1 source; nothing was measured on hardware.

- `lerobot-rollout` (`lerobot/scripts/lerobot_rollout.py::rollout`) builds a
  `RolloutContext` with `build_rollout_context(cfg, shutdown_event)`. It then runs
  `create_strategy(cfg.strategy)` and calls `setup`, `run` and `teardown`. A
  `ProcessSignalHandler` turns SIGINT, SIGTERM, SIGHUP and SIGQUIT into the shutdown event.
  So the v0.2 `command` runner can already stop it cleanly.
- `create_strategy` (`rollout/strategies/factory.py`) is a fixed if-chain over `base`,
  `sentry`, `highlight`, `dagger` and `episodic`. A fieldtrial strategy cannot be
  selected from the CLI without an upstream change; an in-process runner has to call
  `build_rollout_context` and its own `RolloutStrategy` subclass
  (`rollout/strategies/core.py`) directly.
- The context loads the policy once. An in-process runner could keep one context per arm
  loaded and switch arms without reloading, at the cost of GPU memory for every arm.
  Measure the load time and memory of each policy before choosing between that and
  `launch: per_arm`.
- Proposing pluggable strategies upstream (a registry instead of the if-chain) would let
  fieldtrial ship a `lerobot_*` plugin instead.

## v0.3

| Milestone | Scope | Status |
|---|---|---|
| M8 | Anytime-valid comparisons and best-arm selection for serving sweeps; STEP evaluation | done |
| M9 | Reward-model pre-labels with blind human review, Cohen's κ, proxy-assisted (PPI++) intervals | done |
| M10 | In-process LeRobot runner (release 0.3.0) | done |

Limits of M8, to revisit later:

- Anytime-valid stopping supports 2 arms in randomized blocks with one replicate, like
  group-sequential stopping. A non-inferiority margin exists in `fieldtrial.stats`
  (`paired_anytime_test(null=...)`) but not yet in `study.yaml`.
- Best-arm selection needs every surviving arm in every block (`replicates: 1`) and drops an
  arm only on a pairwise confidence sequence; an indifference zone (stop when the leader is
  within ε of the others) could end sweeps sooner.
- STEP (Snyder et al., RSS 2025) is not ported: its code is CC BY-NC 4.0 and may be
  covered by patents (see `docs/stats/step-evaluation.md`).

Limits of M9, to revisit later:

- The Robometer and TOPReward scorers are written against LeRobot 0.6.1's source and its
  own scoring scripts, but have not been run with real weights in fieldtrial's CI (no GPU).
- One score per episode: Robometer's last-frame progress and TOPReward's completion
  probability. Per-frame curves, SARM (which needs a trained checkpoint) and calibration
  of the scores are not used yet.
- Proxy-assisted estimates use the reviewed sample of extra episodes; they estimate the
  success rate of those episodes, not of the scheduled trials. The intervals are
  large-sample (normal) intervals; a betting-based, nonasymptotic version could follow.
- The review page plays the dataset's first camera.

Limits of M10, to revisit later:

- The `lerobot` runner was run against LeRobot 0.6.1 with a simulated robot and small ACT
  checkpoints on a CPU, but not on a real robot, with a GPU, or with RTC inference on a
  policy that supports it.
- It mirrors `build_rollout_context` instead of calling it, because that function loads
  one policy and connects the robot each time; PEFT adapters, `torch.compile`, teleoperated
  resets and DAgger interventions are not supported.
- Preparing an arm loads its policy when the operator presses Start, so the first trial of
  each arm waits for the load. Preloading every arm when the session opens would hide it.
- A candidate upstream proposal: a strategy registry in `lerobot.rollout` (instead of the
  if-chain in `create_strategy`) and a `build_rollout_context` that takes an already
  connected robot, so tools like fieldtrial can reuse it without mirroring it.

## Deferred to the milestone that needs them

- The `openpi`, `capture`, `lerobot` and `rewards` extras are declared together with the
  code that uses them (v0.2 and later).

## Ideas, not scheduled

- Render formulas in the API reference properly. Docstrings use Sphinx `:math:` roles,
  which mkdocstrings shows as plain text; the statistics pages are written in plain
  Unicode.
- Make `import fieldtrial.stats.power as m` return the module: the `power` function,
  exported from `fieldtrial.stats`, shadows the submodule attribute. `from
  fieldtrial.stats.power import ...` works.
- Import scipy lazily in the CLI, so that `fieldtrial --version` starts faster.
- Mark a study `complete` once every slot is done; today its status stays `running`.
- Let CSV imports carry real timestamps (a `started_at` column) instead of laying rows out
  in schedule order.
- Serve uploaded media (clips, photos) in the console and the report; today they are
  stored under `media/` and listed per trial.
- Offline queue in the console: keep labels on the device while the Wi-Fi drops and send
  them when it returns (idempotency keys already make the resend safe).
- Move to `httpx2` for Starlette's TestClient once it is vetted; the tests silence
  Starlette's deprecation warning about `httpx` until then.
- Ship `examples/` in the sdist (or skip `tests/test_examples.py` without it), so the
  whole test suite runs from the sdist alone.
- Test the console on Windows and on real phones (`docs/guides/console-checklist.md`);
  0.1.0 was verified on Linux and in Chromium at phone size.
- `fieldtrial status` could show where a group-sequential study stands (looks done, next
  look due); today that is in the console and `GET /api/v1/studies/{study}/interim`.
- Revisit the docs toolchain. The Material for MkDocs team warns that MkDocs 2.0 removes
  the plugin system, so `docs` pins `mkdocs<2` for now.
