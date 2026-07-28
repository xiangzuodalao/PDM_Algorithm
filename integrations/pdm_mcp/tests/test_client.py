from __future__ import annotations

import httpx
import pytest

from pdm_mcp.client import PdmApiClient
from pdm_mcp.config import Settings
from pdm_mcp.errors import PdmToolError


def _client(handler) -> PdmApiClient:
    settings = Settings(api_base_url="http://pdm.test")
    return PdmApiClient(settings, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_plan_hash_conflict_has_stable_error() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={"detail": "conflict"},
            headers={"X-PDM-Error-Code": "PLAN_HASH_MISMATCH"},
        )

    with pytest.raises(PdmToolError) as caught:
        await _client(handler).request("POST", "/train")
    assert caught.value.payload.code == "PDM_PLAN_CHANGED"
    assert caught.value.payload.http_status == 409


@pytest.mark.asyncio
async def test_server_error_does_not_leak_upstream_detail() -> None:
    secret = "/app/configs/sqlserver_config.json password=hidden"

    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": secret})

    with pytest.raises(PdmToolError) as caught:
        await _client(handler).request("GET", "/healthz")
    assert caught.value.payload.code == "PDM_UPSTREAM_ERROR"
    assert secret not in str(caught.value)


@pytest.mark.asyncio
async def test_training_server_error_is_unknown_outcome() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "failed somewhere"})

    with pytest.raises(PdmToolError) as caught:
        await _client(handler).request("POST", "/train", training_dispatch=True)
    assert caught.value.payload.code == "PDM_TRAIN_OUTCOME_UNKNOWN"
    assert caught.value.payload.retryable is False


@pytest.mark.asyncio
async def test_non_json_training_error_is_unknown_and_not_retryable() -> None:
    async def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(502, text="gateway returned an invalid body")

    with pytest.raises(PdmToolError) as caught:
        await _client(handler).request("POST", "/train", training_dispatch=True)
    assert caught.value.payload.code == "PDM_TRAIN_OUTCOME_UNKNOWN"
    assert caught.value.payload.retryable is False
