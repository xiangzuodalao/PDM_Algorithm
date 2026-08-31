from __future__ import annotations

import asyncio
import hmac
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, StrictInt

from valeo_pdm.db.training_jobs import (
    SqlServerTrainingJobRepository,
    TrainingJobConflict,
    TrainingJobStoreError,
)
from valeo_pdm.paths import configs_dir, resolve_data_path
from valeo_pdm.training.plan import (
    TrainingPlan,
    TrainingPlanError,
    build_training_plan,
    validate_model_info_id,
)
from valeo_pdm.training.run_control import (
    ensure_run_target_available,
    finite_metric,
    mark_interrupted_if_stale,
    read_manifest,
    resolve_safe_run_dir,
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


def predict_informer_api(*args, **kwargs):
    from valeo_pdm.transformer.predict import predict_informer_api as implementation

    return implementation(*args, **kwargs)


def predict_testmodel_api(*args, **kwargs):
    from valeo_pdm.transformer.predict import predict_testmodel_api as implementation

    return implementation(*args, **kwargs)


def predict_autoformer_api(*args, **kwargs):
    from valeo_pdm.transformer.predict import predict_autoformer_api as implementation

    return implementation(*args, **kwargs)


def load_sqlserver_timeseries(*args, **kwargs):
    from valeo_pdm.db.data_reader import load_sqlserver_timeseries as implementation

    return implementation(*args, **kwargs)


def load_postgres_timeseries(*args, **kwargs):
    from valeo_pdm.db.data_reader import load_postgres_timeseries as implementation

    return implementation(*args, **kwargs)


def get_trainer(*args, **kwargs):
    """保留旧模块探针；异步训练实际在 Worker 的 execution 模块中解析 Trainer。"""

    from valeo_pdm.training.registry import get_trainer as implementation

    return implementation(*args, **kwargs)


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


class TrainAcceptedResponse(BaseModel):
    success: bool
    msg: str
    job_id: str
    model_info_id: str
    plan_hash: str
    status: Literal["QUEUED"] = "QUEUED"


class TrainPreviewResponse(BaseModel):
    success: bool
    msg: str
    plan_hash: str
    plan: Dict[str, Any]


class TrainStatusResponse(BaseModel):
    success: bool
    msg: str
    job_id: Optional[str] = None
    model_info_id: str
    plan_hash: Optional[str] = None
    status: Literal["QUEUED", "RUNNING", "SUCCEEDED", "FAILED"]
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
            detail=(f"未找到设备[{request.EquipmentCode}]参数[{request.MeasCode}]的模型配置"),
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
            raise HTTPException(status_code=400, detail="UNSAFE_DATA_PATH") from exc
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
                detail="MODEL_FILE_NOT_FOUND",
            )
    else:
        checkpoint_path = resolve_checkpoint_path(
            request.EquipmentCode, request.MeasCode, model_type
        )
        if not checkpoint_path.exists():
            raise HTTPException(
                status_code=404,
                detail="MODEL_FILE_NOT_FOUND",
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


def _training_job_repository() -> SqlServerTrainingJobRepository:
    return SqlServerTrainingJobRepository.from_json(_sqlserver_config_path())


def _enqueue_training_job(job_id: str) -> None:
    from valeo_pdm.training.tasks import train_model as train_model_task

    train_model_task.apply_async(args=[job_id], task_id=job_id, queue="training")


@router.post(
    "/train",
    response_model=TrainAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="提交异步训练任务",
)
def train_model(request: TrainRequest) -> TrainAcceptedResponse:
    try:
        plan = _request_plan(request)
        _verify_expected_hash(plan, request.ExpectedPlanHash)
        ensure_run_target_available(plan)
    except TrainingPlanError as exc:
        raise _http_plan_error(exc) from exc

    job_id = str(uuid4())
    try:
        repository = _training_job_repository()
        repository.create_queued(job_id, plan)
    except TrainingJobConflict as exc:
        raise HTTPException(
            status_code=409,
            detail="ModelInfoID 对应的训练任务已存在",
            headers={"X-PDM-Error-Code": "MODEL_INFO_ID_CONFLICT"},
        ) from exc
    except TrainingJobStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail="训练任务存储暂不可用",
            headers={"X-PDM-Error-Code": "TRAINING_JOB_STORE_UNAVAILABLE"},
        ) from exc

    try:
        _enqueue_training_job(job_id)
    except Exception as exc:
        try:
            repository.delete_queued(job_id)
        except TrainingJobStoreError:
            pass
        raise HTTPException(
            status_code=503,
            detail="训练队列暂不可用",
            headers={"X-PDM-Error-Code": "TRAINING_QUEUE_UNAVAILABLE"},
        ) from exc

    return TrainAcceptedResponse(
        success=True,
        msg="训练任务已进入队列",
        job_id=job_id,
        model_info_id=plan.model_info_id,
        plan_hash=plan.plan_hash,
        status="QUEUED",
    )


@router.post("/predict", response_model=APIResponse)
async def predict_transformer(request: PredictionRequest):
    try:
        resp = await asyncio.to_thread(_execute_prediction, request)
        return APIResponse(**resp)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="PREDICTION_FAILED") from exc


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
    summary="查询 ModelInfoID 对应的训练任务状态",
)
def get_training_status(
    EquipmentCode: str,
    MeasCode: str,
    ModelInfoID: str,
    ModelType: Optional[str] = None,
):
    try:
        validate_model_info_id(ModelInfoID)
        repository = _training_job_repository()
        job = repository.get_by_identity(EquipmentCode, MeasCode, ModelInfoID)
        if job is not None:
            requested_model_type = str(ModelType or "").strip().lower()
            if requested_model_type and requested_model_type != job.model_type:
                raise TrainingPlanError(
                    "未找到该 ModelInfoID 的训练状态",
                    status_code=404,
                    code="TRAINING_STATUS_NOT_FOUND",
                )
            return TrainStatusResponse(
                success=True,
                msg="获取训练状态成功",
                job_id=job.job_id,
                model_info_id=job.model_info_id,
                plan_hash=job.plan_hash,
                status=job.status,
                created_at=job.created_at.isoformat(),
                started_at=job.started_at.isoformat() if job.started_at else None,
                finished_at=job.finished_at.isoformat() if job.finished_at else None,
                best_val=finite_metric(job.best_val),
                test_loss=finite_metric(job.test_loss),
                error_code=job.error_code,
            )

        # 兼容异步任务表上线前由本地 manifest 记录的训练。
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
            raise TrainingPlanError("训练状态目录无效", status_code=400, code="UNSAFE_OUTPUT_PATH")
        manifest = read_manifest(run_dir)
        if manifest is None:
            raise TrainingPlanError(
                "未找到该 ModelInfoID 的训练状态",
                status_code=404,
                code="TRAINING_STATUS_NOT_FOUND",
            )
        manifest = mark_interrupted_if_stale(run_dir, manifest)
        manifest_status = manifest.get("status")
        status_map = {
            "reserved": "RUNNING",
            "running": "RUNNING",
            "succeeded": "SUCCEEDED",
            "failed": "FAILED",
            "interrupted": "FAILED",
        }
        public_status = status_map.get(manifest_status)
        if public_status is None:
            raise TrainingPlanError("训练状态文件无效", status_code=500, code="INVALID_MANIFEST")
        return TrainStatusResponse(
            success=True,
            msg="获取训练状态成功",
            model_info_id=ModelInfoID,
            plan_hash=manifest.get("plan_hash"),
            status=public_status,
            created_at=manifest.get("created_at"),
            started_at=manifest.get("started_at"),
            finished_at=manifest.get("finished_at"),
            best_val=finite_metric(manifest.get("best_val")),
            test_loss=finite_metric(manifest.get("test_loss")),
            error_code=manifest.get("error_code"),
        )
    except TrainingPlanError as exc:
        raise _http_plan_error(exc) from exc
    except TrainingJobStoreError as exc:
        raise HTTPException(
            status_code=503,
            detail="训练任务存储暂不可用",
            headers={"X-PDM-Error-Code": "TRAINING_JOB_STORE_UNAVAILABLE"},
        ) from exc
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
