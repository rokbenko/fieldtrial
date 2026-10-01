import re
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

import fieldtrial
from fieldtrial.cli.main import app

runner = CliRunner()

# Typer renders help with Rich, which emits ANSI escape codes when it detects CI.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def plain(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_version_comes_from_installed_metadata() -> None:
    assert fieldtrial.__version__ != "0+unknown"


def test_version_flag_prints_the_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.strip() == f"fieldtrial {fieldtrial.__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    output = plain(result.output)
    assert "Usage" in output
    assert "--version" in output


def test_unknown_command_is_a_usage_error() -> None:
    result = runner.invoke(app, ["no-such-command"])
    assert result.exit_code == 2


def test_console_script_is_installed() -> None:
    script = Path(sys.executable).with_name("fieldtrial")
    completed = subprocess.run(
        [str(script), "--version"], capture_output=True, text=True, check=True, timeout=60
    )
    assert completed.stdout.strip() == f"fieldtrial {fieldtrial.__version__}"
