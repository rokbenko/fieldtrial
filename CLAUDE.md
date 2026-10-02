# fieldtrial: notes for Claude Code

Statistically rigorous real-world evaluation for robot policies.
Source of truth: docs/PLAN.md (read it when unsure). Roadmap: docs/ROADMAP.md.

## Commands
- Setup: `uv sync --extra openpi --extra capture --extra lerobot --extra docs` (the dev tools are the default `dev` group; the `rewards` and `lerobot-runner` extras need torch and conflict with it)
- Fast tests: `uv run pytest -m "not slow"`
- All tests: `uv run pytest`
- Lint and format: `uv run ruff check --fix . && uv run ruff format .`
- Types: `uv run mypy src`
- Import contracts: `uv run lint-imports`
- Docs: `uv run mkdocs serve`
- Try it: `uv run fieldtrial demo`

## Architecture rules
- `fieldtrial.stats` is pure: numpy and scipy only, no I/O, no imports from other fieldtrial modules.
- CLI, web and API call `fieldtrial.services`; nothing else touches the DB.
- Every write is one transaction plus one row in the `event` table.
- Never import torch, lerobot or openpi at module import time. Lazy-import inside runners, behind extras.
- All randomness comes from the study seed via `fieldtrial.stats._rng.StableRng`: raw `PCG64(SeedSequence([seed, stream]))` output only, never `Generator` methods, whose streams can change between numpy versions.

## Statistics rules
- Any new or changed stats function needs:
  - a golden test against an independent reference
  - a property test
  - a docs entry with the formula and a reference
- The primary analysis comes from the locked design. Never choose tests after seeing the data.
- Report text goes through `fieldtrial.analysis.wording` only. Never call an arm "better" without a rejected pre-registered test.

## Conventions
- Python ≥3.11, full type hints.
- pydantic v2 for schemas; frozen dataclasses for stats results.
- Conventional commits. Update CHANGELOG.md under "Unreleased" for user-visible changes.
- No network in tests, no telemetry, web assets vendored (no CDN).
- Verify third-party APIs (LeRobot, openpi) against the installed source before using them.
- Out-of-scope ideas go to docs/ROADMAP.md, not into the code.
