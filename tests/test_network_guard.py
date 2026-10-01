import socket

import pytest

BLOCKED = "tests must not use the network"


def test_outbound_connection_is_blocked() -> None:
    with (
        socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock,
        pytest.raises(RuntimeError, match=BLOCKED),
    ):
        sock.connect(("192.0.2.1", 80))  # TEST-NET-1, never routable


def test_name_resolution_is_blocked() -> None:
    with pytest.raises(RuntimeError, match=BLOCKED):
        socket.getaddrinfo("example.com", 443)


def test_loopback_is_allowed() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            pass
