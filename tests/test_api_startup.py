import socket
import sys
import types

import pytest
from fastapi.testclient import TestClient


# The startup contract must remain testable without the production unixODBC shared library.
sys.modules.setdefault("pyodbc", types.SimpleNamespace(connect=lambda *_args, **_kwargs: None))


_ORIGINAL_SOCKET_CONNECT = socket.socket.connect


def _connect_to_loopback_only(sock: socket.socket, address: object) -> None:
    if not isinstance(address, tuple) or not address or address[0] not in {"127.0.0.1", "::1"}:
        raise RuntimeError("应用启动冒烟测试禁止外部网络连接")
    _ORIGINAL_SOCKET_CONNECT(sock, address)


def test_simple_gru_forecast_is_importable() -> None:
    from valeo_pdm.transformer.train_testmodel import SimpleGRUForecast

    assert SimpleGRUForecast.__name__ == "SimpleGRUForecast"


def test_fastapi_app_imports_and_healthz_works(monkeypatch: pytest.MonkeyPatch) -> None:
    from valeo_pdm.api.app import app

    monkeypatch.setattr(socket.socket, "connect", _connect_to_loopback_only)
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
