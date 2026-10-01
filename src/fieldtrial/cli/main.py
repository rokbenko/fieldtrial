"""Entry point of the ``fieldtrial`` command."""

from typing import Annotated

import typer

from fieldtrial import __version__
from fieldtrial.cli import calc

app = typer.Typer(
    name="fieldtrial",
    help="Find out whether your robot policy actually got better.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if not value:
        return
    typer.echo(f"fieldtrial {__version__}")
    raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Find out whether your robot policy actually got better."""


calc.register(app)
