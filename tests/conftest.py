"""Shared test configuration.

Tests never touch the network: every test runs with outbound connections blocked, except to
loopback addresses and Unix sockets.
"""

import socket
from typing import Any

import pytest

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}


class NetworkAccessError(RuntimeError):
    """Raised when a test tries to reach a non-loopback host."""


def _host_of(address: Any) -> str | None:
    if isinstance(address, tuple) and address:
        return str(address[0])
    return None  # Unix socket path


@pytest.fixture(autouse=True)
def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_getaddrinfo = socket.getaddrinfo

    def check(address: Any) -> None:
        host = _host_of(address)
        if host is not None and host not in _LOOPBACK:
            raise NetworkAccessError(f"tests must not use the network (tried {address!r})")

    def connect(self: socket.socket, address: Any) -> None:
        check(address)
        real_connect(self, address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        check(address)
        return real_connect_ex(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        check((host, 0))
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
