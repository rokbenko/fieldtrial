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
| M5 | Crossover rounds, checkpoint ladders (step association, plateau), group-sequential stopping | 0.2.0a1 |
| M6 | Command-template runner and openpi router: real blinding (release 0.2.0a2) | planned |
| M7 | Evaluation-camera capture, rig drift check, LeRobot dataset links (release 0.2.0) | planned |

Limits of M5, to revisit later:

- Crossover designs compare exactly 2 arms; Williams-type crossovers for more arms are
  not supported yet.
- Group-sequential stopping supports 2-arm randomized block designs with one replicate
  (the McNemar score statistic). CMH and multi-arm sequential designs, futility
  boundaries and stopping-adjusted point estimates are not supported yet.
- The ladder's plateau compares each checkpoint with the final one; a margin relative to
  the best checkpoint would need a different procedure.

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
