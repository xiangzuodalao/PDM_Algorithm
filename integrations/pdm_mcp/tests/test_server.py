from __future__ import annotations

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from pdm_mcp import server as server_module
from pdm_mcp.config import HttpSettings, Settings
from pdm_mcp.server import create_server


def _settings() -> Settings:
    return Settings(api_base_url="http://pdm.test", enable_train=False)


def test_server_applies_streamable_http_settings() -> None:
    server = create_server(
        _settings(),
        http_settings=HttpSettings(host="127.0.0.1", port=8765, path="/pdm/mcp"),
    )

    assert server.settings.host == "127.0.0.1"
    assert server.settings.port == 8765
    assert server.settings.streamable_http_path == "/pdm/mcp"
    assert server.settings.stateless_http is False
    assert server.settings.json_response is False
    assert server.settings.transport_security is not None


def test_entrypoints_keep_stdio_and_http_separate(monkeypatch: pytest.MonkeyPatch) -> None:
    transports: list[str] = []

    class FakeServer:
        def run(self, transport: str) -> None:
            transports.append(transport)

    monkeypatch.setattr(server_module, "create_server", lambda *args, **kwargs: FakeServer())

    server_module.main()
    server_module.http_main()

    assert transports == ["stdio", "streamable-http"]


@pytest.mark.asyncio
async def test_streamable_http_protocol_lists_mvp_tools() -> None:
    server = create_server(_settings())
    app = server.streamable_http_app()
    transport = httpx.ASGITransport(app=app)

    async with (
        server.session_manager.run(),
        httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1:8765",
        ) as client,
    ):
        health = await client.get("/healthz")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "server": "pdm-algorithm"}

        async with (
            streamable_http_client(
                "http://127.0.0.1:8765/mcp",
                http_client=client,
            ) as (read_stream, write_stream, _),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()

    assert {tool.name for tool in tools.tools} == {
        "pdm_health_check",
        "pdm_list_models",
        "pdm_check_training_data",
        "pdm_predict",
        "pdm_prepare_training",
        "pdm_train_model",
        "pdm_get_training_status",
    }


@pytest.mark.asyncio
async def test_streamable_http_rejects_untrusted_host() -> None:
    server = create_server(_settings())
    app = server.streamable_http_app()

    async with (
        server.session_manager.run(),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://127.0.0.1:8765",
        ) as client,
    ):
        response = await client.post(
            "/mcp",
            headers={
                "Host": "attacker.example",
                "Accept": "application/json, text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "1"},
                },
            },
        )

    assert response.status_code == 421
