from __future__ import annotations

import os
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from valeo_pdm.paths import configs_dir
from valeo_pdm.training.from_config import save_train_params
from valeo_pdm.training.registry import get_trainer
from valeo_pdm.training.from_config import _format_metrics_desc  # reuse short description formatter
from valeo_pdm.db.train_status import SqlServerTrainStatusUpdater, upload_forecast_image
from valeo_pdm.transformer.artifacts import checkpoint_filename, resolve_checkpoint_path, training_output_dir
from valeo_pdm.transformer.config import get_model_config, list_all_models
from valeo_pdm.transformer.predict import predict_autoformer_api, predict_informer_api, predict_testmodel_api
import math
from valeo_pdm.db.data_reader import load_sqlserver_timeseries, load_postgres_timeseries

MODEL_PREDICT_FUNCS = {
    "informer": predict_informer_api,
    "testmodel": predict_testmodel_api,
    "autoformer": predict_autoformer_api,
}


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


class TrainResponse(BaseModel):
    success: bool
    msg: str
    model_info_id: str
    best_path: Optional[str] = None
    best_val: Optional[float] = None
    test_loss: Optional[float] = None
    run_dir: Optional[str] = None
    train_params: Dict[str, Any] = Field(default_factory=dict)

class DataCheckRequest(BaseModel):
    EquipmentCode: str
    MeasCode: str
    SeqLen: int = 1344      # 模型输入序列长度
    PredLen: int = 672     # 模型预测长度
    DaysBack: int = 365    # 拉取过去多少天的数据
    Source: str = "sqlserver" # 数据源类型


router = APIRouter(prefix="/measPredict", tags=["设备参数预测"])


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str((configs_dir() / "postgres_config.json").resolve())


def _sqlserver_config_path() -> str:
    return os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str((configs_dir() / "sqlserver_config.json").resolve())


def _resolve_csv_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p.resolve())
    return str(p)


PARAM_CASTERS: Dict[str, Any] = {
    "seq_len": int,
    "label_len": int,
    "pred_len": int,
    "batch_size": int,
    "epochs": int,
    "lr": float,
    "d_model": int,
    "n_heads": int,
    "d_ff": int,
    "dropout": float,
    "e_layers": int,
    "d_layers": int,
    "attn_type": str,
    "distil": bool,
    "patience": int,
    "lr_factor": float,
    "lr_patience": int,
    "weight_decay": float,
    "grad_clip": float,
    "stride": int,
    "moving_avg": int,
}

INFORMER_DEFAULTS: Dict[str, Any] = {
    "seq_len": 672,
    "label_len": 192,
    "pred_len": 288,
    "batch_size": 32,
    "epochs": 100,
    "lr": 5e-4,
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 512,
    "dropout": 0.1,
    "e_layers": 2,
    "d_layers": 2,
    "attn_type": "prob",
    "distil": True,
    "patience": 10,
    "lr_factor": 0.5,
    "lr_patience": 5,
    "weight_decay": 0.0,
    "grad_clip": 0.5,
    "stride": 12,
}

AUTOFORMER_DEFAULTS: Dict[str, Any] = {
    "seq_len": 672,
    "label_len": 288,
    "pred_len": 288,
    "batch_size": 128,
    "epochs": 120,
    "lr": 3e-4,
    "moving_avg": 25,
    "d_model": 256,
    "n_heads": 8,
    "d_ff": 512,
    "dropout": 0.1,
    "e_layers": 2,
    "d_layers": 2,
    "patience": 8,
    "lr_factor": 0.5,
    "lr_patience": 4,
    "weight_decay": 1e-4,
    "grad_clip": 0.5,
}


def _parse_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return raw != 0
    if isinstance(raw, str):
        val = raw.strip().lower()
        if val in {"true", "1", "yes", "y", "on"}:
            return True
        if val in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"无法解析布尔值: {raw}")


def _convert_param(name: str, raw: Any) -> Any:
    caster = PARAM_CASTERS.get(name)
    if not caster:
        raise HTTPException(status_code=400, detail=f"不支持的训练参数: {name}")
    try:
        if caster is bool:
            return _parse_bool(raw)
        return caster(raw)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail=f"训练参数[{name}]的值无效: {raw}")


def _build_train_params(param_arr: List[ParamItem]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in param_arr:
        name = item.FieldName
        if name not in PARAM_CASTERS:
            raise HTTPException(status_code=400, detail=f"不支持的训练参数: {name}")
        params[name] = _convert_param(name, item.CurValue)
    return params


@router.post("/train", response_model=TrainResponse, summary="训练模型并按 ModelInfoID 分目录保存")
def train_model(request: TrainRequest):
    try:
        cfg = _postgres_config_path()
        sqlserver_cfg_path = _sqlserver_config_path()
        status_updater = None
        training_started = False
        try:
            status_updater = SqlServerTrainStatusUpdater.from_json(sqlserver_cfg_path)
        except FileNotFoundError:
            status_updater = None
        except ModuleNotFoundError as e:
            # 不阻塞训练，打印提示
            print(f"SQL Server 状态回写缺少依赖: {e}")
        except Exception as e:
            print(f"初始化 SQL Server 状态回写失败: {e}")

        config = get_model_config(request.EquipmentCode, request.MeasCode)
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"未找到设备[{request.EquipmentCode}]参数[{request.MeasCode}]的模型配置",
            )

        req_model_type = request.ModelType or config.get("model_type", "informer")
        model_type = str(req_model_type).lower()
        source = config.get("source", "db")

        if request.DataSource == "sqlserver":
            cfg = _sqlserver_config_path()
            source = "sqlserver"  # 同步更新 source
        elif request.DataSource in ["db", "postgres"]:
            cfg = _postgres_config_path()
            source = request.DataSource  # 同步更新 source (例如 "postgres")
        else:
            # 如果请求中没有传递 DataSource，则回退使用配置中的 source
            if source == "sqlserver":
                cfg = _sqlserver_config_path()
                print(f"命中数据配置: {request.DataSource} == {source} == {cfg}")
            elif source in ["db", "postgres"]:
                cfg = _postgres_config_path()
            else:
                cfg = None  # CSV 不需要数据库配置

        print(f"数据配置: {request.DataSource} == {source} == {cfg}")
        freq = config.get("freq", "15min")
        days_back = int(config.get("days_back", 365))
        data_path = config.get("data_path", "")

        config_train_params = config.get("train_params", {}) or {}
        override_params = _build_train_params(request.ParamArr)
        if model_type == "informer":
            merged_params = {**INFORMER_DEFAULTS, **config_train_params, **override_params}
        elif model_type == "autoformer":
            merged_params = {**AUTOFORMER_DEFAULTS, **config_train_params, **override_params}
        else:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")

        if not request.ModelInfoID:
            raise HTTPException(status_code=400, detail="ModelInfoID 不能为空")

        base_dir = training_output_dir(request.EquipmentCode, request.MeasCode, model_type).resolve()
        run_dir = (base_dir / request.ModelInfoID).resolve()
        save_dir = str(run_dir)
        trainer = get_trainer(model_type)
        if status_updater:
            try:
                status_updater.mark_running(request.ModelInfoID, "Training")
                training_started = True
            except Exception as e:
                print(f"SQL Server 状态回写(训练中)失败: {e}")

        common_kwargs = dict(
            source=source,
            data=_resolve_csv_path(data_path) if source == "csv" else "",
            equipment_code=request.EquipmentCode,
            meas_code=request.MeasCode,
            days_back=days_back,
            freq=freq,
            config=cfg,
        )

        if model_type in {"informer", "testmodel"}:
            distil_raw = merged_params.get("distil", True)
            try:
                distil_bool = distil_raw if isinstance(distil_raw, bool) else _parse_bool(distil_raw)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"训练参数[distil]的值无效: {distil_raw}")

            result = trainer(
                **common_kwargs,
                seq_len=int(merged_params["seq_len"]),
                label_len=int(merged_params["label_len"]),
                pred_len=int(merged_params["pred_len"]),
                batch_size=int(merged_params["batch_size"]),
                epochs=int(merged_params["epochs"]),
                lr=float(merged_params["lr"]),
                save=save_dir,
                d_model=int(merged_params["d_model"]),
                n_heads=int(merged_params.get("n_heads", 4)),
                d_ff=int(merged_params.get("d_ff", 256)),
                dropout=float(merged_params["dropout"]),
                e_layers=int(merged_params.get("e_layers", 2)),
                d_layers=int(merged_params.get("d_layers", 2)),
                attn_type=str(merged_params.get("attn_type", "prob")),
                distil_flag="true" if distil_bool else "false",
                early_stop=True,
                patience=int(merged_params["patience"]),
                lr_sched=True,
                lr_factor=float(merged_params["lr_factor"]),
                lr_patience=int(merged_params["lr_patience"]),
                weight_decay=float(merged_params["weight_decay"]),
                grad_clip=float(merged_params["grad_clip"]),
                stride=int(merged_params["stride"]),
            )
        else:
            if source != "csv":
                raise HTTPException(status_code=400, detail="Autoformer 目前仅支持 CSV 数据源")
            result = trainer(
                **common_kwargs,
                seq_len=int(merged_params["seq_len"]),
                label_len=int(merged_params["label_len"]),
                pred_len=int(merged_params["pred_len"]),
                batch_size=int(merged_params["batch_size"]),
                epochs=int(merged_params["epochs"]),
                lr=float(merged_params["lr"]),
                save=save_dir,
                moving_avg=int(merged_params["moving_avg"]),
                d_model=int(merged_params["d_model"]),
                n_heads=int(merged_params["n_heads"]),
                d_ff=int(merged_params["d_ff"]),
                dropout=float(merged_params["dropout"]),
                e_layers=int(merged_params["e_layers"]),
                d_layers=int(merged_params["d_layers"]),
                early_stop=True,
                patience=int(merged_params["patience"]),
                lr_sched=True,
                lr_factor=float(merged_params["lr_factor"]),
                lr_patience=int(merged_params["lr_patience"]),
                weight_decay=float(merged_params["weight_decay"]),
                grad_clip=float(merged_params["grad_clip"]),
            )

        save_train_params(
            save_dir,
            request.EquipmentCode,
            request.MeasCode,
            model_type,
            source,
            freq,
            days_back,
            data_path,
            merged_params,
        )

        best_path = result.get("best_path")

        if status_updater:
            try:
                desc = _format_metrics_desc(result.get("best_val", "N/A"), result.get("test_loss", "N/A"))

                # 获取本地生成的图片路径
                local_attachment_path = (result.get("plots") or {}).get("forecast")

                # 调用上传接口，获取组装好的 JSON 字符串
                db_attachments = None
                if local_attachment_path:
                    db_attachments = upload_forecast_image(local_attachment_path)

                # 将包含 URL 的 JSON 字符串写入数据库的 attachments 字段
                status_updater.mark_success(request.ModelInfoID, desc, attachments=db_attachments)
            except Exception as e:
                print(f"SQL Server 状态回写(成功)失败: {e}")

        return TrainResponse(
            success=True,
            msg="训练完成",
            model_info_id=request.ModelInfoID,
            best_path=best_path,
            run_dir=save_dir,
            best_val=result.get("best_val"),
            test_loss=result.get("test_loss"),
            train_params=merged_params,
        )
    except HTTPException:
        if status_updater and training_started:
            try:
                status_updater.mark_failed(request.ModelInfoID, "Training failed")
            except Exception as e:
                print(f"SQL Server 状态回写(失败)失败: {e}")
        raise
    except Exception as e:
        if status_updater and training_started:
            try:
                status_updater.mark_failed(request.ModelInfoID, f"Training failed: {e}")
            except Exception as ex:
                print(f"SQL Server 状态回写(失败)失败: {ex}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict", response_model=APIResponse)
async def predict_transformer(request: PredictionRequest):
    if request.ModelInfoID is not None and not request.ModelInfoID.strip():
        request.ModelInfoID = None
    if request.ModelType is not None and not request.ModelType.strip():
        request.ModelType = None
    try:
        cfg = _postgres_config_path()
        config = get_model_config(request.EquipmentCode, request.MeasCode)
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"未找到设备[{request.EquipmentCode}]参数[{request.MeasCode}]的模型配置",
            )

        req_model_type = request.ModelType or config.get("model_type", "informer")
        model_type = str(req_model_type).lower()
        freq = config.get("freq", "15min")
        days_back = int(config.get("days_back", 365))

        predict_func = MODEL_PREDICT_FUNCS.get(model_type)
        if not predict_func:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")

        if request.ModelInfoID:
            run_dir = training_output_dir(request.EquipmentCode, request.MeasCode, model_type) / request.ModelInfoID
            ckpt_path = run_dir / checkpoint_filename(model_type)
            if not ckpt_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到模型文件: {ckpt_path}. 请确认 ModelInfoID 是否正确。",
                )
        else:
            ckpt_path = resolve_checkpoint_path(request.EquipmentCode, request.MeasCode, model_type)
            if not ckpt_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"未找到默认模型文件: {ckpt_path}.",
                )
        resp = await asyncio.to_thread(
            predict_func,
            request.EquipmentCode,
            request.MeasCode,
            freq,
            days_back,
            str(ckpt_path),
            cfg,
        )
        return APIResponse(**resp)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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

@router.post("/checkData", summary="校验训练数据量是否充足")
def check_training_data(request: DataCheckRequest):
    # 1. 获取对应的数据库配置文件路径
    source = request.Source.lower()
    if source == "sqlserver":
        db_config_path = os.getenv("VALEO_PDM_SQLSERVER_CONFIG") or str(
            (configs_dir() / "sqlserver_config.json").resolve())
    elif source in ["db", "postgres"]:
        db_config_path = _postgres_config_path()
    else:
        raise HTTPException(status_code=400, detail="不支持的数据源类型")

    try:
        # 2. 真实拉取数据库，获取总数据量
        if source == "sqlserver":
            df = load_sqlserver_timeseries(db_config_path, request.EquipmentCode, request.MeasCode, request.DaysBack)
        else:
            df = load_postgres_timeseries(db_config_path, request.EquipmentCode, request.MeasCode, request.DaysBack)

        total_data_volume = len(df)

    except ValueError as e:
        # 捕获我们在 data_reader 中写的 df.empty 报错
        total_data_volume = 0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"数据库读取失败: {str(e)}")

    # 3. 算法核心公式计算
    seq_len = request.SeqLen
    pred_len = request.PredLen

    # 理论最低需求行数 (满足 Informer 7:1:2 切分并且能切出至少 1 个样本)
    req_train = (seq_len + pred_len) / 0.7
    req_val = pred_len / 0.1
    required_sample_count = math.ceil(max(req_train, req_val))

    # 总体有效样本数量 (滑动窗口能切出的总样本数)
    effective_sample_count = max(0, total_data_volume - seq_len - pred_len + 1)

    # 欠缺样本数量
    missing_sample_count = max(0, required_sample_count - total_data_volume)

    # 4. 组装返回结果
    return {
        "code": 200,
        "msg": "校验成功" if missing_sample_count == 0 else "数据量不足以支撑训练",
        "data": {
            "TotalDataVolume": total_data_volume,  # 数据总量 (总行数)
            "EffectiveSampleCount": effective_sample_count,  # 有效样本数量 (整段数据能切出的完整样本数)
            "RequiredSampleCount": required_sample_count,  # 需要样本数量 (理论最低安全线)
            "MissingSampleCount": missing_sample_count  # 欠缺样本数量
        }
    }

informer_router = router