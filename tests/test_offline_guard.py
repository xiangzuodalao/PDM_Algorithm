import socket

import pytest


def test_default_suite_blocks_network_connections() -> None:
    with pytest.raises(RuntimeError, match="禁止网络"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.01)
