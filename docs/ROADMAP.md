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

- In-process LeRobot runner.
- Reward-model pre-labeling (Robometer, TOPReward) with human confirmation and agreement
  tracking.
- Proxy-assisted intervals using prediction-powered inference.
- Serving-config sweeps with best-arm identification.
- Near-optimal sequential comparison (evaluate Snyder et al., RSS 2025).

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
- Move the dev tooling from the `dev` extra to a PEP 735 dependency group, so that
  `uv run pytest` works in a fresh clone without `--all-extras`.
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
