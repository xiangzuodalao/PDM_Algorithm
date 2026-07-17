from __future__ import annotations

import importlib
import os
import socket

import pytest


def _network_forbidden(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("默认测试禁止网络和真实数据库连接")


@pytest.fixture(autouse=True)
def isolate_default_tests_from_network(request: pytest.FixtureRequest, monkeypatch) -> None:
    integration = request.node.get_closest_marker("integration_db") is not None
    if integration:
        if os.getenv("VALEO_PDM_ALLOW_READONLY_DB_TESTS") != "1":
            pytest.skip("真实数据库只读集成测试需要显式授权环境变量")
        return

    monkeypatch.setattr(socket, "create_connection", _network_forbidden)
    monkeypatch.setattr(socket.socket, "connect", _network_forbidden)
    for module_name in ("pyodbc", "psycopg2"):
        try:
            database_module = importlib.import_module(module_name)
        except ImportError:
            continue
        monkeypatch.setattr(database_module, "connect", _network_forbidden)
