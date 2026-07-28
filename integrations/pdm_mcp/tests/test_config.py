from __future__ import annotations

import pytest

from pdm_mcp.config import HttpSettings

HTTP_ENV_NAMES = (
    "PDM_MCP_HTTP_HOST",
    "PDM_MCP_HTTP_PORT",
    "PDM_MCP_HTTP_PATH",
    "PDM_MCP_HTTP_ALLOWED_HOSTS",
    "PDM_MCP_HTTP_ALLOWED_ORIGINS",
)


def _clear_http_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in HTTP_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_http_settings_defaults_to_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_http_env(monkeypatch)

    settings = HttpSettings.from_env()

    assert settings == HttpSettings()


def test_http_settings_accepts_explicit_remote_allowlists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_http_env(monkeypatch)
    monkeypatch.setenv("PDM_MCP_HTTP_HOST", "0.0.0.0")
    monkeypatch.setenv("PDM_MCP_HTTP_PORT", "9443")
    monkeypatch.setenv("PDM_MCP_HTTP_PATH", "/pdm/mcp/")
    monkeypatch.setenv("PDM_MCP_HTTP_ALLOWED_HOSTS", "mcp.example.com, mcp.internal:9443")
    monkeypatch.setenv("PDM_MCP_HTTP_ALLOWED_ORIGINS", "https://agent.example.com")

    settings = HttpSettings.from_env()

    assert settings.host == "0.0.0.0"
    assert settings.port == 9443
    assert settings.path == "/pdm/mcp"
    assert settings.allowed_hosts == ("mcp.example.com", "mcp.internal:9443")
    assert settings.allowed_origins == ("https://agent.example.com",)


@pytest.mark.parametrize("port", ["0", "65536", "not-a-port"])
def test_http_settings_rejects_invalid_port(monkeypatch: pytest.MonkeyPatch, port: str) -> None:
    _clear_http_env(monkeypatch)
    monkeypatch.setenv("PDM_MCP_HTTP_PORT", port)

    with pytest.raises(ValueError, match="PDM_MCP_HTTP_PORT"):
        HttpSettings.from_env()


@pytest.mark.parametrize("path", ["mcp", "/", "//mcp", "/mcp?x=1", "/mcp#x"])
def test_http_settings_rejects_invalid_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    _clear_http_env(monkeypatch)
    monkeypatch.setenv("PDM_MCP_HTTP_PATH", path)

    with pytest.raises(ValueError, match="PDM_MCP_HTTP_PATH"):
        HttpSettings.from_env()


def test_http_settings_rejects_remote_bind_without_host_allowlist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_http_env(monkeypatch)
    monkeypatch.setenv("PDM_MCP_HTTP_HOST", "0.0.0.0")

    with pytest.raises(ValueError, match="PDM_MCP_HTTP_ALLOWED_HOSTS"):
        HttpSettings.from_env()
