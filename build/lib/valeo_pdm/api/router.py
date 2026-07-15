from __future__ import annotations

import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from valeo_pdm.paths import configs_dir
from valeo_pdm.transformer.artifacts import resolve_checkpoint_path
from valeo_pdm.transformer.config import get_model_config, list_all_models
from valeo_pdm.transformer.predict import predict_autoformer_api, predict_informer_api


MODEL_PREDICT_FUNCS = {
    "informer": predict_informer_api,
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
    MinDataId: Optional[str] = None
    MaxDataId: Optional[str] = None
    SampleCount: int
    Remark: Optional[str] = ""
    SkippedValues: List[SkippedValue] = []
    Values: List[PredictionValue]


class APIResponse(BaseModel):
    success: bool
    msg: str
    response: PredictionResponseData


router = APIRouter(prefix="/measPredict/transformer", tags=["设备参数预测"])


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str((configs_dir() / "postgres_config.json").resolve())


@router.post("/predict", response_model=APIResponse)
def predict_transformer(request: PredictionRequest):
    try:
        cfg = _postgres_config_path()
        config = get_model_config(request.EquipmentCode, request.MeasCode)
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"未找到设备[{request.EquipmentCode}]参数[{request.MeasCode}]的模型配置",
            )

        model_type = config.get("model_type", "informer")
        freq = config.get("freq", "15min")
        days_back = int(config.get("days_back", 365))

        predict_func = MODEL_PREDICT_FUNCS.get(model_type)
        if not predict_func:
            raise HTTPException(status_code=400, detail=f"不支持的模型类型: {model_type}")

        ckpt_path = resolve_checkpoint_path(request.EquipmentCode, request.MeasCode, model_type)
        resp = predict_func(
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


informer_router = router

