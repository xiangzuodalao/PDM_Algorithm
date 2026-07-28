from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from valeo_pdm.api import router as api_router
from valeo_pdm.api.app import app
from valeo_pdm.training import from_config, run_control
from valeo_pdm.training.plan import TrainingPlanError, build_training_plan
from valeo_pdm.training.run_control import (
    create_training_run,
    ensure_run_target_available,
    mark_interrupted_if_stale,
    read_manifest,
    training_run_lock,
    update_manifest,
)
from valeo_pdm.transformer.artifacts import training_output_dir


BASE_CONFIG: dict[str, Any] = {
    "model_type": "informer",
    "freq": "15min",
    "days_back": 30,
    "source": "db",
    "train_params": {
        "seq_len": 48,
        "label_len": 24,
        "pred_len": 12,
        "batch_size": 16,
        "epochs": 2,
        "lr": 0.0005,
        "d_model": 64,
        "n_heads": 4,
        "d_ff": 128,
        "dropout": 0.1,
        "e_layers": 2,
        "d_layers": 1,
    },
}


def _request(
    model_info_id: str,
    *,
    params: list[dict[str, Any]] | None = None,
    source: str = "db",
    mode: str = "local_only",
    expected_hash: str | None = None,
) -> dict[str, Any]:
    request = {
        "EquipmentCode": "EQ-1",
        "MeasCode": "MEAS-1",
        "ModelInfoID": model_info_id,
        "ParamArr": params or [],
        "DataSource": source,
        "ExecutionMode": mode,
    }
    if expected_hash is not None:
        request["ExpectedPlanHash"] = expected_hash
    return request


def _successful_trainer_result(
    kwargs: dict[str, Any],
    *,
    best_val: float = 0.1,
    test_loss: float = 0.2,
    plots: dict[str, str] | None = None,
) -> dict[str, Any]:
    model_type = "autoformer" if "moving_avg" in kwargs else "informer"
    checkpoint = Path(kwargs["save"]) / f"{model_type}_best.pt"
    checkpoint.write_bytes(b"test checkpoint")
    return {
        "best_path": str(checkpoint),
        "best_val": best_val,
        "test_loss": test_loss,
        "plots": plots or {},
    }


@pytest.fixture
def isolated_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config = deepcopy(BASE_CONFIG)
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    monkeypatch.delenv("VALEO_PDM_REQUIRE_TRAIN_PLAN_HASH", raising=False)
    monkeypatch.setattr(api_router, "get_model_config", lambda *_args: deepcopy(config))
    with TestClient(app) as client:
        yield client, config, tmp_path


def test_preview_is_deterministic_and_has_no_artifact_side_effect(isolated_api) -> None:
    client, _config, root = isolated_api
    first = client.post(
        "/measPredict/train/preview",
        json=_request(
            "mcp-preview-1",
            params=[
                {"FieldName": "epochs", "CurValue": 3},
                {"FieldName": "stride", "CurValue": 2},
            ],
        ),
    )
    second = client.post(
        "/measPredict/train/preview",
        json=_request(
            "mcp-preview-1",
            params=[
                {"FieldName": "stride", "CurValue": 2},
                {"FieldName": "epochs", "CurValue": 3},
            ],
        ),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["plan_hash"] == second.json()["plan_hash"]
    plan = first.json()["plan"]
    assert plan["train_params"]["epochs"] == 3
    assert plan["execution_mode"] == "local_only"
    assert plan["side_effects"] == {
        "local_artifacts": True,
        "sql_status_updates": False,
        "image_upload": False,
    }
    assert not (root / "artifacts").exists()


@pytest.mark.parametrize(
    ("payload", "expected_text"),
    [
        (
            _request(
                "duplicate-1",
                params=[
                    {"FieldName": "epochs", "CurValue": 1},
                    {"FieldName": "epochs", "CurValue": 2},
                ],
            ),
            "训练参数重复",
        ),
        (_request("../escape"), "ModelInfoID"),
        (_request("unknown-source", source="oracle"), "不支持的数据源"),
        (
            _request(
                "too-many-epochs",
                params=[{"FieldName": "epochs", "CurValue": 201}],
            ),
            "epochs",
        ),
        (
            _request(
                "bad-heads",
                params=[{"FieldName": "n_heads", "CurValue": 3}],
            ),
            "整除",
        ),
        (
            _request(
                "wrong-model-param",
                params=[{"FieldName": "moving_avg", "CurValue": 25}],
            ),
            "不支持训练参数",
        ),
    ],
)
def test_preview_rejects_unsafe_or_invalid_plans(isolated_api, payload, expected_text) -> None:
    client, _config, _root = isolated_api
    response = client.post("/measPredict/train/preview", json=payload)
    assert response.status_code == 400
    assert expected_text in response.json()["detail"]


def test_csv_paths_cannot_escape_data_directory(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, config, root = isolated_api
    config.update({"source": "csv", "data_path": "../outside.csv"})
    preview = client.post(
        "/measPredict/train/preview",
        json=_request("unsafe-csv-1", source="csv"),
    )
    assert preview.status_code == 400
    assert "data/" in preview.json()["detail"]

    data = root / "data"
    data.mkdir()
    outside = root / "outside.csv"
    outside.write_text("collect_time,value\n2026-01-01,1\n", encoding="utf-8")
    (data / "escape.csv").symlink_to(outside)
    response = client.post(
        "/measPredict/checkData",
        json={
            "EquipmentCode": "EQ-1",
            "MeasCode": "MEAS-1",
            "SeqLen": 2,
            "LabelLen": 1,
            "PredLen": 1,
            "Stride": 1,
            "Freq": "15min",
            "Source": "csv",
            "DataPath": "data/escape.csv",
        },
    )
    assert response.status_code == 400
    assert "超出 data" in response.json()["detail"]


def test_local_only_train_uses_confirmed_hash_and_status_manifest(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, root = isolated_api
    calls: list[dict[str, Any]] = []

    def trainer(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return _successful_trainer_result(
            kwargs,
            best_val=0.25,
            test_loss=0.5,
            plots={"forecast": "/must/not/upload.png"},
        )

    monkeypatch.setattr(api_router, "get_trainer", lambda _model_type: trainer)
    monkeypatch.setattr(
        api_router.SqlServerTrainStatusUpdater,
        "from_json",
        lambda *_args, **_kwargs: pytest.fail("local_only 不应初始化 SQL 状态回写"),
    )
    monkeypatch.setattr(
        api_router,
        "upload_forecast_image",
        lambda *_args, **_kwargs: pytest.fail("local_only 不应上传图片"),
    )

    payload = _request("mcp-train-1")
    preview = client.post("/measPredict/train/preview", json=payload).json()
    payload["ExpectedPlanHash"] = preview["plan_hash"]
    response = client.post("/measPredict/train", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["plan_hash"] == preview["plan_hash"]
    assert body["status"] == "succeeded"
    assert len(calls) == 1
    assert calls[0]["epochs"] == 2
    run_dir = (
        root / "artifacts" / "checkpoints" / "exp_EQ-1_MEAS-1_informer" / "mcp-train-1"
    )
    manifest = json.loads((run_dir / "training_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "succeeded"
    assert manifest["best_val"] == 0.25

    status = client.get(
        "/measPredict/train/status",
        params={
            "EquipmentCode": "EQ-1",
            "MeasCode": "MEAS-1",
            "ModelInfoID": "mcp-train-1",
        },
    )
    assert status.status_code == 200
    assert status.json()["status"] == "succeeded"
    assert "run_dir" not in status.json()

    conflict = client.post("/measPredict/train", json=payload)
    assert conflict.status_code == 409


def test_hash_drift_and_required_hash_fail_before_creating_run_dir(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, config, root = isolated_api
    payload = _request("hash-guard-1")
    old_hash = client.post("/measPredict/train/preview", json=payload).json()["plan_hash"]
    config["train_params"]["epochs"] = 3
    payload["ExpectedPlanHash"] = old_hash

    drift = client.post("/measPredict/train", json=payload)
    assert drift.status_code == 409
    assert "重新 preview" in drift.json()["detail"]

    monkeypatch.setenv("VALEO_PDM_REQUIRE_TRAIN_PLAN_HASH", "1")
    required = client.post("/measPredict/train", json=_request("hash-required-1"))
    assert required.status_code == 428
    assert not (
        root
        / "artifacts"
        / "checkpoints"
        / "exp_EQ-1_MEAS-1_informer"
        / "hash-required-1"
    ).exists()


def test_hash_is_rechecked_under_lock_before_directory_creation(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, root = isolated_api
    payload = _request("locked-hash-1")
    expected = client.post("/measPredict/train/preview", json=payload).json()["plan_hash"]
    original = deepcopy(BASE_CONFIG)
    changed = deepcopy(BASE_CONFIG)
    changed["train_params"]["epochs"] = 3
    calls = 0

    def changing_config(*_args: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return deepcopy(original if calls == 1 else changed)

    monkeypatch.setattr(api_router, "get_model_config", changing_config)
    payload["ExpectedPlanHash"] = expected
    response = client.post("/measPredict/train", json=payload)
    assert response.status_code == 409
    assert calls == 2
    assert not (
        root
        / "artifacts"
        / "checkpoints"
        / "exp_EQ-1_MEAS-1_informer"
        / "locked-hash-1"
    ).exists()


def test_failed_training_is_manifested_without_leaking_internal_error(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, _root = isolated_api

    def fail_trainer(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("secret connection string")

    monkeypatch.setattr(api_router, "get_trainer", lambda _model_type: fail_trainer)
    payload = _request("failed-1")
    payload["ExpectedPlanHash"] = client.post(
        "/measPredict/train/preview", json=payload
    ).json()["plan_hash"]
    response = client.post("/measPredict/train", json=payload)
    assert response.status_code == 500
    assert response.json()["detail"] == "训练执行失败"
    assert "secret" not in response.text

    status = client.get(
        "/measPredict/train/status",
        params={
            "EquipmentCode": "EQ-1",
            "MeasCode": "MEAS-1",
            "ModelInfoID": "failed-1",
        },
    )
    assert status.status_code == 200
    assert status.json()["status"] == "failed"
    assert status.json()["error_code"] == "TRAINING_FAILED"


def test_missing_checkpoint_marks_training_failed(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, _root = isolated_api
    monkeypatch.setattr(
        api_router,
        "get_trainer",
        lambda _model_type: lambda **_kwargs: {"best_val": 0.1, "test_loss": 0.2},
    )
    payload = _request("missing-checkpoint-1")
    payload["ExpectedPlanHash"] = client.post(
        "/measPredict/train/preview", json=payload
    ).json()["plan_hash"]
    response = client.post("/measPredict/train", json=payload)
    assert response.status_code == 500
    status = client.get(
        "/measPredict/train/status",
        params={
            "EquipmentCode": "EQ-1",
            "MeasCode": "MEAS-1",
            "ModelInfoID": "missing-checkpoint-1",
        },
    )
    assert status.json()["status"] == "failed"


def test_predict_rejects_model_info_id_path_traversal(isolated_api) -> None:
    client, _config, _root = isolated_api
    response = client.post(
        "/measPredict/predict",
        json={
            "EquipmentCode": "EQ-1",
            "MeasCode": "MEAS-1",
            "ModelInfoID": "../other-model",
        },
    )
    assert response.status_code == 400
    assert "ModelInfoID" in response.json()["detail"]


def test_platform_mode_keeps_status_and_forecast_upload_behavior(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, _root = isolated_api
    events: list[str] = []

    class StatusUpdater:
        def mark_running(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("running")

        def mark_success(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("success")

        def mark_failed(self, *_args: Any, **_kwargs: Any) -> None:
            events.append("failed")

    monkeypatch.setattr(
        api_router.SqlServerTrainStatusUpdater,
        "from_json",
        lambda *_args, **_kwargs: StatusUpdater(),
    )
    monkeypatch.setattr(
        api_router,
        "upload_forecast_image",
        lambda path: events.append(f"upload:{path}") or "[]",
    )
    monkeypatch.setattr(
        api_router,
        "get_trainer",
        lambda _model_type: lambda **kwargs: _successful_trainer_result(
            kwargs, plots={"forecast": "/tmp/forecast.png"}
        ),
    )
    payload = _request("platform-1", mode="platform")
    preview = client.post("/measPredict/train/preview", json=payload).json()
    assert preview["plan"]["side_effects"]["sql_status_updates"] is True
    payload["ExpectedPlanHash"] = preview["plan_hash"]
    response = client.post("/measPredict/train", json=payload)
    assert response.status_code == 200
    assert events == ["running", "upload:/tmp/forecast.png", "success"]


def test_legacy_platform_request_without_hash_remains_compatible(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, _root = isolated_api
    monkeypatch.setattr(
        api_router.SqlServerTrainStatusUpdater,
        "from_json",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        api_router,
        "get_trainer",
        lambda _model_type: lambda **kwargs: _successful_trainer_result(kwargs),
    )
    payload = _request("legacy-no-hash-1")
    payload.pop("ExecutionMode")
    response = client.post("/measPredict/train", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


def test_autoformer_training_keeps_model_specific_arguments(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, config, _root = isolated_api
    config.update({"model_type": "autoformer", "source": "csv", "data_path": "data/test.csv"})
    captured: dict[str, Any] = {}

    def trainer(**kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs)
        return _successful_trainer_result(kwargs)

    monkeypatch.setattr(api_router, "get_trainer", lambda _model_type: trainer)
    payload = _request("autoformer-1", source="csv")
    payload["ModelType"] = "autoformer"
    payload["ExpectedPlanHash"] = client.post(
        "/measPredict/train/preview", json=payload
    ).json()["plan_hash"]
    response = client.post("/measPredict/train", json=payload)
    assert response.status_code == 200
    assert captured["source"] == "csv"
    assert captured["moving_avg"] == 25
    assert "distil_flag" not in captured

    rejected = _request(
        "autoformer-wrong-param",
        source="csv",
        params=[{"FieldName": "attn_type", "CurValue": "full"}],
    )
    rejected["ModelType"] = "autoformer"
    response = client.post("/measPredict/train/preview", json=rejected)
    assert response.status_code == 400
    assert "不支持训练参数" in response.json()["detail"]


def test_check_data_exposes_usable_in_both_compatibility_locations(
    isolated_api, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _config, _root = isolated_api
    quality = {
        "input_rows": 20,
        "filtered_rows": 20,
        "dropped_invalid_time_rows": 0,
        "dropped_invalid_value_rows": 0,
        "valid_rows_before_resample": 20,
        "aggregated_rows": 20,
        "rows_after_resample": 20,
    }
    monkeypatch.setattr(api_router, "load_timeseries_file", lambda *_a, **_k: (list(range(20)), quality))
    response = client.post(
        "/measPredict/checkData",
        json={
            "EquipmentCode": "EQ-1",
            "MeasCode": "MEAS-1",
            "SeqLen": 2,
            "LabelLen": 1,
            "PredLen": 1,
            "Stride": 1,
            "Freq": "15min",
            "Source": "csv",
            "DataPath": "data/test.csv",
        },
    )
    assert response.status_code == 200
    assert response.json()["usable"] is True
    assert response.json()["data"]["Usable"] is True


def test_non_finite_training_parameter_is_rejected() -> None:
    with pytest.raises(TrainingPlanError, match="lr"):
        build_training_plan(
            equipment_code="EQ-1",
            meas_code="MEAS-1",
            model_info_id="nan-1",
            requested_model_type="informer",
            param_items=[("lr", float("nan"))],
            requested_source="db",
            execution_mode="local_only",
            model_config=BASE_CONFIG,
        )


def test_run_lock_is_cross_instance_non_blocking_and_symlinks_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    plan = build_training_plan(
        equipment_code="EQ-1",
        meas_code="MEAS-1",
        model_info_id="lock-1",
        requested_model_type="informer",
        param_items=[],
        requested_source="db",
        execution_mode="local_only",
        model_config=BASE_CONFIG,
    )
    with training_run_lock(plan):
        with pytest.raises(TrainingPlanError, match="正在执行"):
            with training_run_lock(plan):
                pass

    base = training_output_dir("EQ-1", "MEAS-1", "informer")
    base.mkdir(parents=True)
    (base / "lock-1").symlink_to(base / "missing-target")
    with pytest.raises(TrainingPlanError):
        ensure_run_target_available(plan)


def test_cli_model_info_id_uses_shared_exclusive_run_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    monkeypatch.setattr(from_config, "get_model_config", lambda *_args: deepcopy(BASE_CONFIG))
    save_dirs: list[str] = []

    def fake_impl(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        save_dirs.append(kwargs.get("_save_dir_override") or _args[5])
        return {"best_val": 0.1, "test_loss": 0.2}

    monkeypatch.setattr(from_config, "_train_from_config_impl", fake_impl)
    result = from_config.train_from_config("EQ-1", "MEAS-1", "cli-run-1")
    assert result["best_val"] == 0.1
    assert save_dirs[0].endswith("/exp_EQ-1_MEAS-1_informer/cli-run-1")
    manifest = json.loads(
        (Path(save_dirs[0]) / "training_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "succeeded"
    with pytest.raises(TrainingPlanError):
        from_config.train_from_config("EQ-1", "MEAS-1", "cli-run-1")


def test_cli_locked_config_snapshot_is_passed_to_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    first = deepcopy(BASE_CONFIG)
    second = deepcopy(BASE_CONFIG)
    second["train_params"]["epochs"] = 3
    reads = iter((first, second))
    monkeypatch.setattr(from_config, "get_model_config", lambda *_args: deepcopy(next(reads)))
    captured: dict[str, Any] = {}

    def fake_impl(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(kwargs["_config_override"])
        return {"best_val": 0.1, "test_loss": 0.2}

    monkeypatch.setattr(from_config, "_train_from_config_impl", fake_impl)
    from_config.train_from_config("EQ-1", "MEAS-1", "cli-snapshot-1")
    assert captured["train_params"]["epochs"] == 3


def test_stale_running_manifest_becomes_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    plan = build_training_plan(
        equipment_code="EQ-1",
        meas_code="MEAS-1",
        model_info_id="stale-1",
        requested_model_type="informer",
        param_items=[],
        requested_source="db",
        execution_mode="local_only",
        model_config=BASE_CONFIG,
    )
    with training_run_lock(plan):
        run_dir = create_training_run(plan)
    manifest = update_manifest(run_dir, status="running", pid=987654321)

    def process_missing(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr("valeo_pdm.training.run_control.os.kill", process_missing)
    refreshed = mark_interrupted_if_stale(run_dir, manifest)
    assert refreshed["status"] == "interrupted"
    assert read_manifest(run_dir)["error_code"] == "TRAINING_PROCESS_EXITED"


def test_reused_pid_with_new_process_identity_becomes_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    plan = build_training_plan(
        equipment_code="EQ-1",
        meas_code="MEAS-1",
        model_info_id="reused-pid-1",
        requested_model_type="informer",
        param_items=[],
        requested_source="db",
        execution_mode="local_only",
        model_config=BASE_CONFIG,
    )
    with training_run_lock(plan):
        run_dir = create_training_run(plan)
    manifest = update_manifest(
        run_dir, status="running", pid=42, process_identity="old-boot:100"
    )
    monkeypatch.setattr(run_control, "_process_identity", lambda _pid: "new-boot:200")
    refreshed = mark_interrupted_if_stale(run_dir, manifest)
    assert refreshed["status"] == "interrupted"


def test_initial_manifest_failure_does_not_consume_model_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("VALEO_PDM_ROOT", str(tmp_path))
    plan = build_training_plan(
        equipment_code="EQ-1",
        meas_code="MEAS-1",
        model_info_id="manifest-failure-1",
        requested_model_type="informer",
        param_items=[],
        requested_source="db",
        execution_mode="local_only",
        model_config=BASE_CONFIG,
    )
    monkeypatch.setattr(
        run_control,
        "write_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    with training_run_lock(plan):
        with pytest.raises(OSError, match="disk unavailable"):
            create_training_run(plan)
    base = training_output_dir("EQ-1", "MEAS-1", "informer")
    assert not (base / "manifest-failure-1").exists()
    assert not list(base.glob(".reserve-*"))
