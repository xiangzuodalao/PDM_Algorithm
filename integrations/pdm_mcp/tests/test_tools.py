from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import httpx
import pytest

from pdm_mcp.client import PdmApiClient
from pdm_mcp.config import Settings
from pdm_mcp.errors import PdmToolError
from pdm_mcp.preview_store import PreviewStore
from pdm_mcp.schemas import TrainParams
from pdm_mcp.server import create_server
from pdm_mcp.tools import PdmTools


def _settings(*, enable_train: bool = True) -> Settings:
    return Settings(
        api_base_url="http://pdm.test",
        api_timeout_seconds=1,
        train_timeout_seconds=1,
        enable_train=enable_train,
        preview_ttl_seconds=900,
    )


def _plan(*, model_info_id: str = "demo-001", source: str = "db") -> dict[str, Any]:
    return {
        "version": 1,
        "equipment_code": "EQ-1",
        "meas_code": "M-1",
        "model_info_id": model_info_id,
        "model_type": "informer",
        "source": source,
        "freq": "15min",
        "days_back": 365,
        "data_path": "data/demo.csv" if source == "csv" else "/secret/ignored.csv",
        "train_params": {
            "seq_len": 48,
            "label_len": 24,
            "pred_len": 12,
            "stride": 2,
            "batch_size": 16,
            "epochs": 1,
            "lr": 0.001,
            "d_model": 32,
            "n_heads": 4,
            "d_ff": 64,
            "dropout": 0.1,
            "e_layers": 1,
            "d_layers": 1,
            "attn_type": "full",
            "distil": False,
            "patience": 2,
            "lr_factor": 0.5,
            "lr_patience": 1,
            "weight_decay": 0.0,
            "grad_clip": 0.5,
        },
        "execution_mode": "local_only",
        "side_effects": {
            "local_artifacts": True,
            "sql_status_updates": False,
            "image_upload": False,
        },
        "output_policy": {"exclusive_new_directory": True},
        "internal_path": "/must/not/leak",
    }


def _plan_hash(plan: dict[str, Any]) -> str:
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


PLAN_HASH = _plan_hash(_plan())


def _data_check(*, usable: bool = True) -> dict[str, Any]:
    windows = 4 if usable else 0
    return {
        "code": 200,
        "msg": "校验成功",
        "usable": usable,
        "data": {
            "Usable": usable,
            "RawDataVolume": 1000,
            "TotalDataVolume": 900,
            "EffectiveSampleCount": 50,
            "RequiredSampleCount": 70,
            "MissingSampleCount": 0,
            "TrainWindowCount": 40 if usable else 50,
            "ValidationWindowCount": windows,
            "TestWindowCount": windows,
            "DataQuality": {
                "input_rows": 1000,
                "filtered_rows": 900,
                "dropped_invalid_time_rows": 50,
                "dropped_invalid_value_rows": 50,
                "valid_rows_before_resample": 900,
                "aggregated_rows": 900,
                "rows_after_resample": 900,
                "internal_path": "/must/not/leak.csv",
            },
        },
    }


def _json_body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


@pytest.mark.asyncio
async def test_prepare_then_train_is_single_use_and_hides_paths() -> None:
    requests: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = _json_body(request) if request.content else {}
        requests.append((request.url.path, body))
        if request.url.path.endswith("/train/preview"):
            return httpx.Response(
                200,
                json={"success": True, "msg": "ok", "plan_hash": PLAN_HASH, "plan": _plan()},
            )
        if request.url.path.endswith("/checkData"):
            return httpx.Response(200, json=_data_check())
        if request.url.path.endswith("/train"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "msg": "训练完成",
                    "model_info_id": "demo-001",
                    "status": "succeeded",
                    "best_path": "/app/artifacts/private.pt",
                    "run_dir": "/app/artifacts/private",
                    "best_val": 0.12,
                    "test_loss": 0.23,
                    "plan_hash": PLAN_HASH,
                    "train_params": {"epochs": 1},
                },
            )
        raise AssertionError(request.url)

    settings = _settings()
    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(handler)),
    )
    prepared = await tools.prepare_training(
        "EQ-1",
        "M-1",
        "demo-001",
        "db",
        TrainParams(seq_len=48, label_len=24, pred_len=12, epochs=1),
    )
    assert prepared["usable"] is True
    assert prepared["preview_id"]
    assert prepared["plan"]["data_path"] == ""
    assert "internal_path" not in prepared["plan"]
    assert "internal_path" not in prepared["data_check"]["data_quality"]
    preview_request = requests[0][1]
    assert "ExpectedPlanHash" not in preview_request

    result = await tools.train_model(prepared["preview_id"])
    assert result["status"] == "succeeded"
    assert "best_path" not in result
    assert "run_dir" not in result
    train_request = requests[-1][1]
    assert train_request["ExpectedPlanHash"] == PLAN_HASH

    with pytest.raises(PdmToolError) as caught:
        await tools.train_model(prepared["preview_id"])
    assert caught.value.payload.code == "PDM_PREVIEW_ALREADY_USED"
    assert [path for path, _ in requests].count("/measPredict/train") == 1


@pytest.mark.asyncio
async def test_unusable_data_does_not_issue_preview_id() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/train/preview"):
            return httpx.Response(
                200,
                json={"success": True, "msg": "ok", "plan_hash": PLAN_HASH, "plan": _plan()},
            )
        return httpx.Response(200, json=_data_check(usable=False))

    settings = _settings()
    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(handler)),
    )
    result = await tools.prepare_training("EQ-1", "M-1", "demo-001", "db")
    assert result["usable"] is False
    assert result["preview_id"] is None
    assert result["expires_at"] is None


@pytest.mark.asyncio
async def test_training_transport_failure_has_unknown_outcome_and_no_retry() -> None:
    train_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal train_calls
        if request.url.path.endswith("/train/preview"):
            return httpx.Response(
                200,
                json={"success": True, "msg": "ok", "plan_hash": PLAN_HASH, "plan": _plan()},
            )
        if request.url.path.endswith("/checkData"):
            return httpx.Response(200, json=_data_check())
        train_calls += 1
        raise httpx.ReadTimeout("unknown", request=request)

    settings = _settings()
    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(handler)),
    )
    prepared = await tools.prepare_training("EQ-1", "M-1", "demo-001", "db")
    with pytest.raises(PdmToolError) as caught:
        await tools.train_model(prepared["preview_id"])
    assert caught.value.payload.code == "PDM_TRAIN_OUTCOME_UNKNOWN"
    with pytest.raises(PdmToolError) as consumed:
        await tools.train_model(prepared["preview_id"])
    assert consumed.value.payload.code == "PDM_PREVIEW_ALREADY_USED"
    assert train_calls == 1


@pytest.mark.asyncio
async def test_malformed_success_after_training_has_unknown_outcome() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/train/preview"):
            return httpx.Response(
                200,
                json={"success": True, "plan_hash": PLAN_HASH, "plan": _plan()},
            )
        if request.url.path.endswith("/checkData"):
            return httpx.Response(200, json=_data_check())
        return httpx.Response(
            200,
            json={"success": True, "status": "succeeded", "model_info_id": "wrong-id"},
        )

    settings = _settings()
    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(handler)),
    )
    prepared = await tools.prepare_training("EQ-1", "M-1", "demo-001", "db")
    with pytest.raises(PdmToolError) as caught:
        await tools.train_model(prepared["preview_id"])
    assert caught.value.payload.code == "PDM_TRAIN_OUTCOME_UNKNOWN"


@pytest.mark.asyncio
async def test_prepare_rejects_mismatched_source_and_hash() -> None:
    plan = _plan(source="csv")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "plan_hash": _plan_hash(plan), "plan": plan},
        )

    settings = _settings()
    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(PdmToolError) as caught:
        await tools.prepare_training("EQ-1", "M-1", "demo-001", "db")
    assert caught.value.payload.code == "PDM_INVALID_RESPONSE"

    valid_plan = _plan()

    def bad_hash_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"success": True, "plan_hash": "b" * 64, "plan": valid_plan},
        )

    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(bad_hash_handler)),
    )
    with pytest.raises(PdmToolError) as bad_hash:
        await tools.prepare_training("EQ-1", "M-1", "demo-001", "db")
    assert bad_hash.value.payload.code == "PDM_INVALID_RESPONSE"


@pytest.mark.asyncio
async def test_predict_samples_full_result_and_computes_statistics() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert set(_json_body(request)) == {"EquipmentCode", "MeasCode"}
        values = [
            {"Seq": index, "XAxis": f"t{index}", "Value": str(index), "Unit": "A"}
            for index in range(10)
        ]
        return httpx.Response(
            200,
            json={
                "success": True,
                "msg": "预测成功",
                "response": {"SampleCount": 99, "Remark": "", "Values": values},
            },
        )

    settings = _settings()
    tools = PdmTools(
        settings,
        PdmApiClient(settings, transport=httpx.MockTransport(handler)),
    )
    result = await tools.predict("EQ-1", "M-1", max_points=3)
    assert [item["seq"] for item in result["values"]] == [0, 4, 9]
    assert result["statistics"]["mean"] == 4.5
    assert result["truncated"] is True


@pytest.mark.asyncio
async def test_csv_path_is_restricted_to_data_directory() -> None:
    settings = _settings()
    tools = PdmTools(
        settings, PdmApiClient(settings, transport=httpx.MockTransport(lambda _: None))
    )
    with pytest.raises(PdmToolError) as caught:
        await tools.check_training_data("EQ-1", "M-1", "csv", data_path="../secret.csv")
    assert caught.value.payload.code == "PDM_UNSAFE_DATA_PATH"


@pytest.mark.asyncio
async def test_data_check_rejects_excessive_ranges_before_http() -> None:
    settings = _settings()
    tools = PdmTools(
        settings, PdmApiClient(settings, transport=httpx.MockTransport(lambda _: None))
    )
    with pytest.raises(PdmToolError) as windows:
        await tools.check_training_data("EQ-1", "M-1", "db", seq_len=10001)
    assert windows.value.payload.code == "PDM_INVALID_ARGUMENT"
    with pytest.raises(PdmToolError) as days:
        await tools.check_training_data("EQ-1", "M-1", "db", days_back=3651)
    assert days.value.payload.code == "PDM_INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_preview_store_consumes_once_under_concurrency() -> None:
    store = PreviewStore(ttl_seconds=900)
    preview_id, _ = await store.create({"request": {"value": 1}})

    async def consume() -> str:
        try:
            await store.consume(preview_id)
        except PdmToolError as exc:
            return exc.payload.code
        return "consumed"

    results = await asyncio.gather(consume(), consume())
    assert sorted(results) == ["PDM_PREVIEW_ALREADY_USED", "consumed"]


@pytest.mark.asyncio
async def test_preview_store_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = [100.0]
    monkeypatch.setattr("pdm_mcp.preview_store.time.monotonic", lambda: clock[0])
    store = PreviewStore(ttl_seconds=1)
    preview_id, _ = await store.create({"request": {}})
    clock[0] += 2
    with pytest.raises(PdmToolError) as caught:
        await store.consume(preview_id)
    assert caught.value.payload.code == "PDM_PREVIEW_EXPIRED"


@pytest.mark.asyncio
async def test_server_exposes_exact_mvp_tool_set() -> None:
    server = create_server(_settings(enable_train=False))
    names = {tool.name for tool in await server.list_tools()}
    assert names == {
        "pdm_health_check",
        "pdm_list_models",
        "pdm_check_training_data",
        "pdm_predict",
        "pdm_prepare_training",
        "pdm_train_model",
        "pdm_get_training_status",
    }
