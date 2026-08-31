from __future__ import annotations

import asyncio
import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, StrictInt

from valeo_pdm.paths import configs_dir, resolve_data_path
from valeo_pdm.training.from_config import _format_metrics_desc  # reuse short description formatter
from valeo_pdm.training.from_config import save_train_params
from valeo_pdm.training.plan import (
    TrainingPlan,
    TrainingPlanError,
    build_training_plan,
    validate_model_info_id,
)
from valeo_pdm.db.train_status import SqlServerTrainStatusUpdater, upload_forecast_image
from valeo_pdm.training.registry import get_trainer
from valeo_pdm.training.run_control import (
    create_training_run,
    ensure_run_target_available,
    finite_metric,
    mark_interrupted_if_stale,
    read_manifest,
    resolve_safe_run_dir,
    training_run_lock,
    update_manifest,
)
from valeo_pdm.transformer.artifacts import (
    checkpoint_filename,
    resolve_checkpoint_path,
)
from valeo_pdm.transformer.config import get_model_config, list_all_models
from valeo_pdm.transformer.data import (
    clean_and_resample_timeseries,
    load_timeseries_file,
    minimum_rows_for_usable_window_splits,
    validate_window_params,
    window_split_counts,
)
from valeo_pdm.transformer.predict import (
    predict_autoformer_api,
    predict_informer_api,
    predict_testmodel_api,
)
from valeo_pdm.db.data_reader import load_sqlserver_timeseries, load_postgres_timeseries

MODEL_PREDICT_FUNCS = {
    "informer": predict_informer_api,
    "testmodel": predict_testmodel_api,
    "autoformer": predict_autoformer_api,
}
# 场景 onboarding 工具通过 AST 读取该常量，必须保留为本文件中的字面量赋值。
API_TRAIN_MODEL_TYPES = frozenset({"informer", "autoformer"})


@dataclass(frozen=True)
class PredictionPlan:
    equipment_code: str
    meas_code: str
    model_type: str
    freq: str
    days_back: int
    source: str
    data_path: str
    config_path: str
    checkpoint_path: Path
    predict_func: Callable[..., dict[str, Any]]


class DynamicParams(BaseModel):
    DynamicParam_1: Optional[str] = ""
    DynamicParam_2: Optional[str] = ""
    DynamicParam_3: Optional[str] = ""


class HistoryDataItem(BaseModel):
    id: str
    timestamp: str
    value: float
    unit: Optional[str] = ""


class PredictionRequest(BaseModel):
    EquipmentCode: str
    MeasCode: str
    ModelInfoID: Optional[str] = None
    ModelType: Optional[str] = None
    DynamicParameters: DynamicParams = DynamicParams()
    HistoryData: List[HistoryDataItem] = []


class SkippedValue(BaseModel):
    DataId: str
    XAxis: Optional[str] = None
    Value: Optional[str] = None
    Unit: Optional[str] = ""


class PredictionValue(BaseModel):
    Seq: int
    XAxis: Optional[str] = None
    Value: str
    Unit: Optional[str] = ""


class PredictionResponseData(BaseModel):
    # MinDataId: Optional[str] = None
    # MaxDataId: Optional[str] = None
    SampleCount: int
    Remark: Optional[str] = ""
    # SkippedValues: List[SkippedValue] = []
    Values: List[PredictionValue]


class APIResponse(BaseModel):
    success: bool
    msg: str
    response: PredictionResponseData


class ParamItem(BaseModel):
    FieldName: str
    CurValue: Any


class TrainRequest(BaseModel):
    EquipmentCode: str
    MeasCode: str
    ModelInfoID: str
    ModelType: Optional[str] = None
    ParamArr: List[ParamItem]
    DataSource: str
    ExpectedPlanHash: Optional[str] = None
    ExecutionMode: Literal["platform", "local_only"] = "platform"


class TrainResponse(BaseModel):
    success: bool
    msg: str
    model_info_id: str
    best_path: Optional[str] = None
    best_val: Optional[float] = None
    test_loss: Optional[float] = None
    run_dir: Optional[str] = None
    train_params: Dict[str, Any] = Field(default_factory=dict)
    plan_hash: str
    status: Literal["succeeded"] = "succeeded"


class TrainPreviewResponse(BaseModel):
    success: bool
    msg: str
    plan_hash: str
    plan: Dict[str, Any]


class TrainStatusResponse(BaseModel):
    success: bool
    msg: str
    model_info_id: str
    status: Literal["reserved", "running", "succeeded", "failed", "interrupted"]
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    best_val: Optional[float] = None
    test_loss: Optional[float] = None
    error_code: Optional[str] = None


class DataCheckRequest(BaseModel):
    EquipmentCode: str
    MeasCode: str
    SeqLen: StrictInt = 1344
    LabelLen: StrictInt = 336
    PredLen: StrictInt = 672
    Stride: StrictInt = 1
    Freq: str = "15min"
    DaysBack: int = 365
    Source: str = "sqlserver"
    DataPath: Optional[str] = None


router = APIRouter(prefix="/measPredict", tags=["设备参数预测"])


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str(
        (configs_dir() / "postgres_config.json").resolve()
    )


def _sqlserver_config_path() -> str:
    return os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str(
        (configs_dir() / "sqlserver_config.json").resolve()
    )


def _resolve_csv_path(path_str: str) -> str:
    return str(resolve_data_path(path_str))


def _resolve_prediction_plan(request: PredictionRequest) -> PredictionPlan:
    model_info_id = (
        request.ModelInfoID
        if request.ModelInfoID is not None and request.ModelInfoID.strip()
        else None
    )
    requested_model_type = (
        request.ModelType if request.ModelType is not None and request.ModelType.strip() else None
    )

    config = get_model_config(request.EquipmentCode, request.MeasCode)
    if not config:
        raise HTTPException(
            status_code=404,
            detail=(
                f"未找到设备[{request.EquipmentCode}]"
                f"参数[{request.MeasCode}]的模型配置"
            ),
        )

    model_type = str(requested_model_type or config.get("model_type", "informer")).lower()
    freq = str(config.get("freq", "15min"))
    days_back = int(config.get("days_back", 365))
    source = str(config.get("source", "db")).strip().lower()
    data_path = str(config.get("data_path", ""))

    if source == "sqlserver":
        config_path = _sqlserver_config_path()
    elif source in {"db", "postgres"}:
        config_path = _postgres_config_path()
    elif source == "csv":
        config_path = ""
        try:
            data_path = _resolve_csv_path(data_path)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    else:
        raise HTTPException(status_code=400, detail=f"不支持的数据源: {source}")

    predict_func = MODEL_PREDICT_FUNCS.get(model_type)
    if predict_func is None:
        raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")
    if model_type == "autoformer" and source != "csv":
        raise HTTPException(status_code=400, detail="Autoformer 仅支持 CSV 数据源")

    if model_info_id:
        try:
            validate_model_info_id(model_info_id)
            identity = TrainingPlan(
                equipment_code=request.EquipmentCode,
                meas_code=request.MeasCode,
                model_info_id=model_info_id,
                model_type=model_type,
                source=source,
                freq=freq,
                days_back=days_back,
                data_path=data_path,
                train_params={},
                execution_mode="local_only",
            )
            checkpoint_path = resolve_safe_run_dir(identity) / checkpoint_filename(model_type)
        except TrainingPlanError as exc:
            raise _http_plan_error(exc) from exc
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404,
                detail=(
                    f"未找到模型文件: {checkpoint_path}. "
                    "请确认 ModelInfoID 是否正确。"
                ),
            )
    else:
        checkpoint_path = resolve_checkpoint_path(
            request.EquipmentCode, request.MeasCode, model_type
        )
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"未找到默认模型文件: {checkpoint_path}.",
            )

    return PredictionPlan(
        equipment_code=request.EquipmentCode,
        meas_code=request.MeasCode,
        model_type=model_type,
        freq=freq,
        days_back=days_back,
        source=source,
        data_path=data_path,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        predict_func=predict_func,
    )


def _execute_prediction(request: PredictionRequest) -> dict[str, Any]:
    plan = _resolve_prediction_plan(request)
    return plan.predict_func(
        plan.equipment_code,
        plan.meas_code,
        plan.freq,
        plan.days_back,
        str(plan.checkpoint_path),
        plan.config_path,
        source=plan.source,
        data_path=plan.data_path,
    )


def _request_plan(request: TrainRequest) -> TrainingPlan:
    config = get_model_config(request.EquipmentCode, request.MeasCode)
    return build_training_plan(
        equipment_code=request.EquipmentCode,
        meas_code=request.MeasCode,
        model_info_id=request.ModelInfoID,
        requested_model_type=request.ModelType,
        param_items=((item.FieldName, item.CurValue) for item in request.ParamArr),
        requested_source=request.DataSource,
        execution_mode=request.ExecutionMode,
        model_config=config,
    )


def _http_plan_error(exc: TrainingPlanError) -> HTTPException:
    return HTTPException(
        status_code=exc.status_code,
        detail=str(exc),
        headers={"X-PDM-Error-Code": exc.code},
    )


def _hash_required() -> bool:
    return os.getenv("VALEO_PDM_REQUIRE_TRAIN_PLAN_HASH", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _verify_expected_hash(plan: TrainingPlan, expected_hash: str | None) -> None:
    if not expected_hash:
        if _hash_required():
            raise TrainingPlanError(
                "训练前必须先调用 preview 并提交 ExpectedPlanHash",
                status_code=428,
                code="PLAN_HASH_REQUIRED",
            )
        return
    if not hmac.compare_digest(plan.plan_hash, expected_hash.strip().lower()):
        raise TrainingPlanError(
            "训练计划已变化，请重新 preview 并确认",
            status_code=409,
            code="PLAN_HASH_MISMATCH",
        )


def _database_config_for_plan(plan: TrainingPlan) -> str | None:
    if plan.source == "sqlserver":
        return _sqlserver_config_path()
    if plan.source in {"db", "postgres"}:
        return _postgres_config_path()
    return None


def _invoke_trainer(plan: TrainingPlan, save_dir: str) -> dict[str, Any]:
    params = plan.train_params
    trainer = get_trainer(plan.model_type)
    common_kwargs = dict(
        source=plan.source,
        data=_resolve_csv_path(plan.data_path) if plan.source == "csv" else "",
        equipment_code=plan.equipment_code,
        meas_code=plan.meas_code,
        days_back=plan.days_back,
        freq=plan.freq,
        config=_database_config_for_plan(plan),
    )
    if plan.model_type == "informer":
        return trainer(
            **common_kwargs,
            seq_len=params["seq_len"],
            label_len=params["label_len"],
            pred_len=params["pred_len"],
            batch_size=params["batch_size"],
            epochs=params["epochs"],
            lr=params["lr"],
            save=save_dir,
            d_model=params["d_model"],
            n_heads=params["n_heads"],
            d_ff=params["d_ff"],
            dropout=params["dropout"],
            e_layers=params["e_layers"],
            d_layers=params["d_layers"],
            attn_type=params["attn_type"],
            distil_flag="true" if params["distil"] else "false",
            early_stop=True,
            patience=params["patience"],
            lr_sched=True,
            lr_factor=params["lr_factor"],
            lr_patience=params["lr_patience"],
            weight_decay=params["weight_decay"],
            grad_clip=params["grad_clip"],
            stride=params["stride"],
        )
    return trainer(
        **common_kwargs,
        seq_len=params["seq_len"],
        label_len=params["label_len"],
        pred_len=params["pred_len"],
        batch_size=params["batch_size"],
        epochs=params["epochs"],
        lr=params["lr"],
        save=save_dir,
        moving_avg=params["moving_avg"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        early_stop=True,
        patience=params["patience"],
        lr_sched=True,
        lr_factor=params["lr_factor"],
        lr_patience=params["lr_patience"],
        weight_decay=params["weight_decay"],
        grad_clip=params["grad_clip"],
        stride=params["stride"],
    )


def _validated_training_checkpoint(
    plan: TrainingPlan, run_dir: Path, result: dict[str, Any]
) -> Path:
    raw = result.get("best_path")
    if not isinstance(raw, (str, os.PathLike)):
        raise RuntimeError("trainer did not return best_path")
    declared = Path(raw)
    if declared.is_symlink():
        raise RuntimeError("trainer checkpoint cannot be a symlink")
    try:
        resolved = declared.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError("trainer checkpoint does not exist") from exc
    expected = (run_dir / checkpoint_filename(plan.model_type)).resolve(strict=False)
    if resolved != expected or not resolved.is_file():
        raise RuntimeError("trainer checkpoint is outside the reserved run directory")
    return resolved


@router.post(
    "/train/preview",
    response_model=TrainPreviewResponse,
    summary="解析并预览无副作用的训练计划",
)
def preview_training(request: TrainRequest):
    try:
        plan = _request_plan(request)
        ensure_run_target_available(plan)
    except TrainingPlanError as exc:
        raise _http_plan_error(exc) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="训练计划解析失败") from exc
    return TrainPreviewResponse(
        success=True,
        msg="训练计划预览成功",
        plan_hash=plan.plan_hash,
        plan=plan.as_dict(),
    )


@router.post("/train", response_model=TrainResponse, summary="训练模型并按 ModelInfoID 分目录保存")
def train_model(request: TrainRequest):
    return _execute_train_model(request)


def _platform_status_updater(plan: TrainingPlan) -> SqlServerTrainStatusUpdater | None:
    if plan.execution_mode != "platform":
        return None
    try:
        return SqlServerTrainStatusUpdater.from_json(_sqlserver_config_path())
    except (FileNotFoundError, ModuleNotFoundError):
        return None
    except Exception:
        print("初始化 SQL Server 训练状态回写失败，已继续本地训练。")
        return None


def _execute_train_model(request: TrainRequest) -> TrainResponse:
    run_dir = None
    status_updater = None
    training_started = False
    try:
        initial_plan = _request_plan(request)
        _verify_expected_hash(initial_plan, request.ExpectedPlanHash)

        with training_run_lock(initial_plan):
            # 锁内重新读取 DB/YAML 配置，避免 preview 后配置变化造成 TOCTOU。
            plan = _request_plan(request)
            _verify_expected_hash(plan, request.ExpectedPlanHash)
            run_dir = create_training_run(plan)
            update_manifest(
                run_dir,
                status="running",
                started_at=datetime.now(UTC).isoformat(),
                pid=os.getpid(),
            )
            training_started = True
            status_updater = _platform_status_updater(plan)
            if status_updater is not None:
                try:
                    status_updater.mark_running(plan.model_info_id, "Training")
                except Exception:
                    print("SQL Server 状态回写(训练中)失败，已继续本地训练。")

            try:
                result = _invoke_trainer(plan, str(run_dir))
                if not isinstance(result, dict):
                    raise TypeError("trainer result must be a mapping")
                best_path = _validated_training_checkpoint(plan, run_dir, result)
                save_train_params(
                    str(run_dir),
                    plan.equipment_code,
                    plan.meas_code,
                    plan.model_type,
                    plan.source,
                    plan.freq,
                    plan.days_back,
                    plan.data_path,
                    plan.train_params,
                )
            except Exception:
                update_manifest(
                    run_dir,
                    status="failed",
                    finished_at=datetime.now(UTC).isoformat(),
                    error_code="TRAINING_FAILED",
                )
                if status_updater is not None:
                    try:
                        status_updater.mark_failed(plan.model_info_id, "Training failed")
                    except Exception:
                        print("SQL Server 状态回写(失败)失败。")
                raise

            best_val = finite_metric(result.get("best_val"))
            test_loss = finite_metric(result.get("test_loss"))
            if status_updater is not None:
                try:
                    description = _format_metrics_desc(best_val, test_loss)
                    forecast_path = (result.get("plots") or {}).get("forecast")
                    attachments = upload_forecast_image(forecast_path) if forecast_path else None
                    status_updater.mark_success(
                        plan.model_info_id, description, attachments=attachments
                    )
                except Exception:
                    print("SQL Server 状态回写或预测图上传失败，本地训练结果仍然有效。")

            update_manifest(
                run_dir,
                status="succeeded",
                finished_at=datetime.now(UTC).isoformat(),
                best_val=best_val,
                test_loss=test_loss,
                error_code=None,
            )
            return TrainResponse(
                success=True,
                msg="训练完成",
                model_info_id=plan.model_info_id,
                best_path=str(best_path),
                run_dir=str(run_dir),
                best_val=best_val,
                test_loss=test_loss,
                train_params=plan.train_params,
                plan_hash=plan.plan_hash,
                status="succeeded",
            )
    except TrainingPlanError as exc:
        raise _http_plan_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        if run_dir is not None and training_started:
            try:
                manifest = read_manifest(run_dir) or {}
                if manifest.get("status") not in {"failed", "succeeded"}:
                    update_manifest(
                        run_dir,
                        status="failed",
                        finished_at=datetime.now(UTC).isoformat(),
                        error_code="TRAINING_FAILED",
                    )
            except Exception:
                pass
        raise HTTPException(status_code=500, detail="训练执行失败") from exc


@router.post("/predict", response_model=APIResponse)
async def predict_transformer(request: PredictionRequest):
    try:
        resp = await asyncio.to_thread(_execute_prediction, request)
        return APIResponse(**resp)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/models", summary="列出所有已注册的模型配置")
def list_models():
    try:
        models = list_all_models()
        result = []
        for equipment_code, params in models.items():
            for meas_code, config in params.items():
                result.append(
                    {
                        "equipment_code": equipment_code,
                        "meas_code": meas_code,
                        "model_type": config.get("model_type", "informer"),
                        "freq": config.get("freq"),
                        "source": config.get("source", "db"),
                    }
                )
        return {"success": True, "msg": "获取模型列表成功", "models": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/train/status",
    response_model=TrainStatusResponse,
    summary="查询 ModelInfoID 对应的本地训练状态",
)
def get_training_status(
    EquipmentCode: str,
    MeasCode: str,
    ModelInfoID: str,
    ModelType: Optional[str] = None,
):
    try:
        validate_model_info_id(ModelInfoID)
        config = get_model_config(EquipmentCode, MeasCode)
        if config is None:
            raise TrainingPlanError(
                f"未找到设备[{EquipmentCode}]参数[{MeasCode}]的模型配置",
                status_code=404,
                code="SCENARIO_NOT_FOUND",
            )
        model_type = str(ModelType or config.get("model_type", "informer")).strip().lower()
        if model_type not in API_TRAIN_MODEL_TYPES:
            raise TrainingPlanError(
                f"不支持的模型类型: {model_type}", code="UNSUPPORTED_MODEL_TYPE"
            )
        identity = TrainingPlan(
            equipment_code=EquipmentCode,
            meas_code=MeasCode,
            model_info_id=ModelInfoID,
            model_type=model_type,
            source="db",
            freq="",
            days_back=1,
            data_path="",
            train_params={},
            execution_mode="local_only",
        )
        run_dir = resolve_safe_run_dir(identity)
        if run_dir.is_symlink():
            raise TrainingPlanError(
                "训练状态目录无效", status_code=400, code="UNSAFE_OUTPUT_PATH"
            )
        manifest = read_manifest(run_dir)
        if manifest is None:
            raise TrainingPlanError(
                "未找到该 ModelInfoID 的训练状态",
                status_code=404,
                code="TRAINING_STATUS_NOT_FOUND",
            )
        manifest = mark_interrupted_if_stale(run_dir, manifest)
        status = manifest.get("status")
        if status not in {"reserved", "running", "succeeded", "failed", "interrupted"}:
            raise TrainingPlanError(
                "训练状态文件无效", status_code=500, code="INVALID_MANIFEST"
            )
        return TrainStatusResponse(
            success=True,
            msg="获取训练状态成功",
            model_info_id=ModelInfoID,
            status=status,
            created_at=manifest.get("created_at"),
            started_at=manifest.get("started_at"),
            finished_at=manifest.get("finished_at"),
            best_val=finite_metric(manifest.get("best_val")),
            test_loss=finite_metric(manifest.get("test_loss")),
            error_code=manifest.get("error_code"),
        )
    except TrainingPlanError as exc:
        raise _http_plan_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="训练状态查询失败") from exc


@router.post("/checkData", summary="校验训练数据量是否充足")
def check_training_data(request: DataCheckRequest):
    try:
        validate_window_params(
            request.SeqLen,
            request.LabelLen,
            request.PredLen,
            request.Stride,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    source = request.Source.strip().lower()
    empty_quality = {
        "input_rows": 0,
        "filtered_rows": 0,
        "dropped_invalid_time_rows": 0,
        "dropped_invalid_value_rows": 0,
        "valid_rows_before_resample": 0,
        "aggregated_rows": 0,
        "rows_after_resample": 0,
    }

    if source == "csv":
        if not request.DataPath:
            raise HTTPException(status_code=400, detail="CSV 数据校验缺少 DataPath")
        try:
            cleaned, quality = load_timeseries_file(
                _resolve_csv_path(request.DataPath),
                request.Freq,
                equipment_code=request.EquipmentCode,
                meas_code=request.MeasCode,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f"CSV 数据无效: {exc}") from exc
    elif source in {"db", "postgres", "sqlserver"}:
        db_config_path = (
            _sqlserver_config_path() if source == "sqlserver" else _postgres_config_path()
        )
        try:
            if source == "sqlserver":
                raw = load_sqlserver_timeseries(
                    db_config_path,
                    request.EquipmentCode,
                    request.MeasCode,
                    request.DaysBack,
                )
            else:
                raw = load_postgres_timeseries(
                    db_config_path,
                    request.EquipmentCode,
                    request.MeasCode,
                    request.DaysBack,
                )
        except ValueError:
            raw = None
        except Exception as exc:
            raise HTTPException(status_code=500, detail="数据库读取失败") from exc

        if raw is None:
            cleaned = None
            quality = empty_quality
        else:
            try:
                cleaned, quality = clean_and_resample_timeseries(
                    raw,
                    request.Freq,
                    equipment_code=request.EquipmentCode,
                    meas_code=request.MeasCode,
                )
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"时序数据无效: {exc}") from exc
    else:
        raise HTTPException(status_code=400, detail="不支持的数据源类型")

    rows_after_resample = 0 if cleaned is None else len(cleaned)
    counts = window_split_counts(
        rows_after_resample,
        request.SeqLen,
        request.LabelLen,
        request.PredLen,
        request.Stride,
    )
    required_rows = minimum_rows_for_usable_window_splits(
        request.SeqLen,
        request.LabelLen,
        request.PredLen,
        request.Stride,
    )
    missing_rows = max(0, required_rows - rows_after_resample)
    usable = (
        min(
            counts["train_windows"],
            counts["val_windows"],
            counts["test_windows"],
        )
        >= 1
    )

    return {
        "code": 200,
        "msg": "校验成功" if usable else "数据量不足以支撑训练",
        "usable": usable,
        "data": {
            "Usable": usable,
            "RawDataVolume": quality["input_rows"],
            "TotalDataVolume": rows_after_resample,
            "EffectiveSampleCount": counts["windows"],
            "RequiredSampleCount": required_rows,
            "MissingSampleCount": missing_rows,
            "TrainWindowCount": counts["train_windows"],
            "ValidationWindowCount": counts["val_windows"],
            "TestWindowCount": counts["test_windows"],
            "DataQuality": quality,
        },
    }


informer_router = router
