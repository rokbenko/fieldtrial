# Roadmap

The plan, its milestones and the decision log are in
[the project plan](PLAN.md). This page tracks progress and anything that is out of scope
for the current milestone.

## v0.1

| Milestone | Scope | Status |
|---|---|---|
| M0 | Bootstrap: packaging, CI, docs skeleton, `fieldtrial --version` | done |
| M1 | Statistics core and calculator CLI (release 0.1.0a1) | released 2026-10-02 |
| M2 | Study design, storage and analysis (no UI) | done, unreleased |
| M3 | Operator console, REST API and Python client | done, unreleased |
| M4 | Reports, docs and the Dream Machines re-analysis (release 0.1.0) | next |

## v0.2

- Command-template runner: spawns your rollout command per arm, which makes real blinding
  possible.
- openpi router: blinded A/B testing for websocket policy servers.
- Evaluation-camera capture (`capture` extra).
- Links between trials and LeRobot dataset episodes.
- Checkpoint ladders: trend test and plateau detection.
- Crossover-rounds design for tasks where scene state carries over between trials.
- Rig drift check against reference images.
- Group-sequential stopping.

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
- Revisit the docs toolchain. The Material for MkDocs team warns that MkDocs 2.0 removes
  the plugin system, so `docs` pins `mkdocs<2` for now.
