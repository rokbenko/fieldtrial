"""Architecture rules from docs/PLAN.md, section 6.

import-linter (``uv run lint-imports``) forbids specific imports. These tests add what it
can't express: an allow-list for ``fieldtrial.stats``, and a check that importing the
package never loads heavy optional dependencies, even when they are installed.
"""

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import fieldtrial

PACKAGE_DIR = Path(fieldtrial.__file__).parent
STATS_ALLOWED_THIRD_PARTY = {"numpy", "scipy"}

# Must never be loaded by `import fieldtrial` or the CLI (lazy-import inside runners only).
HEAVY_OR_TEST_ONLY = (
    "av",
    "cv2",
    "lerobot",
    "openpi",
    "openpi_client",
    "statsmodels",
    "torch",
    "websockets",
)


def _imported_top_level_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path}: use absolute imports"
            assert node.module is not None
            names.add(node.module)
    return names


def test_stats_imports_only_stdlib_numpy_scipy_and_itself() -> None:
    files = sorted((PACKAGE_DIR / "stats").rglob("*.py"))
    assert files, "fieldtrial.stats has no modules"
    for path in files:
        for name in _imported_top_level_names(path):
            top = name.split(".")[0]
            if top == "fieldtrial":
                assert name == "fieldtrial.stats" or name.startswith("fieldtrial.stats."), (
                    f"{path.name} imports {name}; fieldtrial.stats must stay self-contained"
                )
            else:
                assert top in sys.stdlib_module_names or top in STATS_ALLOWED_THIRD_PARTY, (
                    f"{path.name} imports {name}; fieldtrial.stats may only use numpy and scipy"
                )


def test_importing_fieldtrial_never_loads_heavy_dependencies() -> None:
    # A meta-path hook makes any attempt to import a forbidden package fail loudly, which
    # also catches the case where the package is installed in the environment.
    script = textwrap.dedent(
        f"""
        import importlib.abc
        import sys

        FORBIDDEN = {HEAVY_OR_TEST_ONLY!r}

        class Guard(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in FORBIDDEN:
                    raise ImportError(f"forbidden import at import time: {{name}}")
                return None

        sys.meta_path.insert(0, Guard())

        import fieldtrial
        import fieldtrial.cli.main
        import fieldtrial.stats

        print("ok")
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_stats_does_not_load_cli_or_ui_libraries() -> None:
    script = textwrap.dedent(
        """
        import sys

        import fieldtrial.stats

        loaded = sorted(
            name for name in ("typer", "rich", "pydantic", "sqlalchemy", "fastapi", "matplotlib")
            if name in sys.modules
        )
        print(",".join(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=120
    )
    assert completed.stdout.strip() == ""
