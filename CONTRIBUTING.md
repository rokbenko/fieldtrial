# Contributing to fieldtrial

Thanks for helping. fieldtrial's job is to tell people whether their robot policy actually
got better, so correctness matters more than features. A wrong p-value is worse than no
p-value.

## Set up

You need [uv](https://docs.astral.sh/uv/) 0.12.21 or newer.

```bash
git clone https://github.com/rokbenko/fieldtrial
cd fieldtrial
uv sync --all-extras
uv run pre-commit install
```

## Checks

Run these before every commit. CI runs the same ones on Python 3.11, 3.12 and 3.13, on
Linux and macOS.

```bash
uv run ruff check --fix . && uv run ruff format .   # lint and format
uv run mypy src                                     # types (strict)
uv run lint-imports                                 # architecture contracts
uv run pytest -m "not slow"                         # fast tests
uv run pytest                                       # everything, including slow statistical tests
uv run mkdocs build --strict                        # docs
```

## Rules

- **Statistics.** Any new or changed function in `fieldtrial.stats` needs all three of:
  - a golden-value test against an independent reference implementation
  - a property-based test (Hypothesis)
  - a docs entry with the formula and a reference

  Write the tests first.
- **Architecture.**
  - `fieldtrial.stats` depends only on numpy and scipy and imports no other fieldtrial
    package.
  - Nothing imports torch, lerobot or openpi at module import time.
  - `uv run lint-imports` and `tests/test_architecture.py` enforce this.
- **No network in tests.** A fixture blocks every non-loopback connection.
- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/), for
  example `feat(stats): add Wilson interval`.
- **Changelog.** User-visible changes get a line under "Unreleased" in `CHANGELOG.md`.
- **Scope.** Ideas outside the current milestone go into `docs/ROADMAP.md`.

The project plan and its decision log live in `docs/PLAN.md`.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
