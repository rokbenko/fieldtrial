"""``fieldtrial serve``: the operator console and REST API."""

import io
import secrets
import socket
from pathlib import Path
from typing import Annotated

import typer

from fieldtrial.cli.study import FolderArg, _errors


def lan_address() -> str:
    """The address of this machine on the local network (nothing is sent), or 127.0.0.1."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: routing lookup only, nothing is sent
        address = str(probe.getsockname()[0])
    except OSError:
        address = "127.0.0.1"
    finally:
        probe.close()
    return address


def qr_text(url: str) -> str:
    """A QR code for ``url`` drawn with text characters, for the terminal."""
    import qrcode

    code = qrcode.QRCode(border=2)
    code.add_data(url)
    code.make(fit=True)
    out = io.StringIO()
    code.print_ascii(out=out, invert=True)
    return out.getvalue()


def console_url(folder: Path, *, lan: bool, port: int, token: str | None) -> str:
    """The address to open on a phone or laptop."""
    host = lan_address() if lan else "127.0.0.1"
    url = f"http://{host}:{port}/"
    return f"{url}?token={token}" if token else url


@_errors
def serve(
    folder: FolderArg,
    lan: Annotated[
        bool,
        typer.Option("--lan", help="Serve on the local network (needs the printed token)."),
    ] = False,
    port: Annotated[int, typer.Option("--port", help="TCP port.")] = 8765,
) -> None:
    """Run the operator console and REST API for a study (or a folder of studies)."""
    import uvicorn

    from fieldtrial.web.app import create_app

    token = secrets.token_urlsafe(16) if lan else None
    app = create_app(folder, lan_token=token)
    url = console_url(Path(folder), lan=lan, port=port, token=token)
    typer.echo(f"fieldtrial console: {url}")
    if lan:
        typer.echo(qr_text(url))
        typer.echo("Anyone with this URL can record trials. Stop the server with Ctrl+C.")
    uvicorn.run(app, host="0.0.0.0" if lan else "127.0.0.1", port=port, log_level="warning")


def register(app: typer.Typer) -> None:
    """Add ``serve`` to the main app."""
    app.command("serve")(serve)
