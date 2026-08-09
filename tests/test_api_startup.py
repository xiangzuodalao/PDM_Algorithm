import socket
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


_ORIGINAL_SOCKET_CONNECT = socket.socket.connect


def _connect_to_loopback_only(sock: socket.socket, address: object) -> None:
    if not isinstance(address, tuple) or not address or address[0] not in {"127.0.0.1", "::1"}:
        raise RuntimeError("应用启动冒烟测试禁止外部网络连接")
    _ORIGINAL_SOCKET_CONNECT(sock, address)


def test_api_app_is_importable_without_loading_a_model_stack() -> None:
    from valeo_pdm.api.app import app

    assert app is not None


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


def test_readyz_is_safe_when_no_prediction_runtime_is_configured() -> None:
    from valeo_pdm.api.app import app

    with TestClient(app) as client:
        response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json() == {"status": "not_ready"}


def test_app_import_isolated_from_pyodbc_and_test_modules_do_not_inject_it() -> None:
    """Eager DB imports or test shims would make API startup depend on a native driver."""
    root = Path(__file__).resolve().parents[1]
    program = """
import builtins
import runpy
import sys

original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == 'pyodbc':
        raise ImportError('pyodbc deliberately blocked')
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
import valeo_pdm.api.app
assert 'pyodbc' not in sys.modules
runpy.run_path('tests/test_api_startup.py')
runpy.run_path('tests/test_prediction_v2_api.py')
assert 'pyodbc' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=root, text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
