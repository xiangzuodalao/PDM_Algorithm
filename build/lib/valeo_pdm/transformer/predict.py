from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch

from valeo_pdm.db.data_reader import load_postgres_timeseries
from valeo_pdm.paths import configs_dir
from valeo_pdm.transformer.artifacts import resolve_checkpoint_path
from valeo_pdm.transformer.models.autoformer import Autoformer
from valeo_pdm.transformer.models.informer import Informer
from valeo_pdm.transformer.train_informer import normalize_columns, parse_time, resample_group


def _clean_residuals(residuals: np.ndarray) -> np.ndarray:
    r = np.asarray(residuals, dtype=np.float32)
    if r.size == 0:
        return r
    q1 = np.quantile(r, 0.25)
    q3 = np.quantile(r, 0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return r[(r >= low) & (r <= high)]


def _generate_future_residuals(residuals: np.ndarray, length: int, strength: float) -> np.ndarray:
    if residuals.size == 0:
        return np.zeros(length, dtype=np.float32)
    mu = float(residuals.mean())
    sigma = float(residuals.std() + 1e-6)
    noise = np.random.normal(loc=mu, scale=sigma, size=length).astype(np.float32)
    return noise * float(strength)


def _freq_desc(freq: str) -> str:
    f = str(freq).lower().strip()
    if f.endswith("min"):
        try:
            n = int(f[:-3])
            return f"{n}分钟级"
        except Exception:
            return "分钟级"
    if f == "t":
        return "分钟级"
    if f == "5t":
        return "5分钟级"
    if f == "h":
        return "小时级"
    return f


def build_api_response_informer(
    equipment_code: str, meas_code: str, freq: str, future_times, future_values, original_history_data: pd.DataFrame
) -> Dict[str, Any]:
    values = [
        {
            "Seq": i + 1,
            "XAxis": pd.Timestamp(future_times[i]).strftime("%Y-%m-%d %H:%M:%S"),
            "Value": f"{float(future_values[i]):.9f}",
            "Unit": "",
        }
        for i in range(len(future_values))
    ]
    remark = f"设备: {equipment_code}, 模型: informer, 预测粒度: {_freq_desc(freq)}"
    return {
        "success": True,
        "msg": f"设备[{equipment_code}]参数[{meas_code}]预测成功",
        "response": {
            "MinDataId": None,
            "MaxDataId": None,
            "SampleCount": int(len(original_history_data)),
            "Remark": remark,
            "SkippedValues": [],
            "Values": values,
        },
    }


def build_api_response_autoformer(
    equipment_code: str, meas_code: str, freq: str, future_times, future_values, original_history_data: pd.DataFrame
) -> Dict[str, Any]:
    values = [
        {
            "Seq": i + 1,
            "XAxis": pd.Timestamp(future_times[i]).strftime("%Y-%m-%d %H:%M:%S"),
            "Value": f"{float(future_values[i]):.9f}",
            "Unit": "",
        }
        for i in range(len(future_values))
    ]
    remark = f"设备: {equipment_code}, 模型: autoformer, 预测粒度: {_freq_desc(freq)}"
    return {
        "success": True,
        "msg": f"设备[{equipment_code}]参数[{meas_code}]预测成功",
        "response": {
            "MinDataId": None,
            "MaxDataId": None,
            "SampleCount": int(len(original_history_data)),
            "Remark": remark,
            "SkippedValues": [],
            "Values": values,
        },
    }


def _postgres_config_path() -> str:
    return os.getenv("VALEO_PDM_POSTGRES_CONFIG") or str((configs_dir() / "postgres_config.json").resolve())


def predict_informer_api(
    equipment_code: str,
    meas_code: str,
    freq: str,
    days_back: int,
    ckpt_path: str,
    config_path: str,
    add_residual: bool = True,
    residual_strength: float = 0.8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.load(ckpt_path, map_location=device)
    m = float(c["mean"])
    s = float(c["std"])
    a = c.get("args", {})
    sl = int(a.get("seq_len", 672))
    pl = int(a.get("pred_len", 288))
    model = Informer(
        seq_len=sl,
        label_len=int(a.get("label_len", 192)),
        pred_len=pl,
        d_model=int(a.get("d_model", 256)),
        n_heads=int(a.get("n_heads", 8)),
        d_ff=int(a.get("d_ff", 512)),
        dropout=float(a.get("dropout", 0.1)),
        e_layers=int(a.get("e_layers", 2)),
        d_layers=int(a.get("d_layers", 2)),
        attn_type=str(a.get("attn_type", "prob")),
        distil=(str(a.get("distil", "true")).lower() == "true"),
    ).to(device)
    model.load_state_dict(c["model"])
    model.eval()

    df = load_postgres_timeseries(config_path, equipment_code, meas_code, days_back=days_back)
    df = normalize_columns(df)
    df = parse_time(df)
    g = df[df["meas_code"] == meas_code] if "meas_code" in df.columns else df
    sd = resample_group(g, freq=freq)
    vals = pd.to_numeric(sd["value"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if len(vals) < sl:
        return build_api_response_informer(equipment_code, meas_code, freq, [], [], df)
    x = vals[-sl:]
    x_norm = (x - m) / s
    x_t = torch.from_numpy(np.asarray(x_norm)[None, :, None]).to(device)

    with torch.no_grad():
        y_hat = model(x_t)
    y_pred_den = y_hat[0].detach().cpu().numpy() * s + m

    if add_residual and len(vals) > sl:
        hist_len = min(len(vals) - sl, sl * 2)
        raw_residuals = []
        for i in range(hist_len):
            start_idx = len(vals) - sl - hist_len + i
            if start_idx < 0:
                continue
            hist_x = vals[start_idx : start_idx + sl]
            hist_x_norm = (hist_x - m) / s
            hist_x_t = torch.from_numpy(hist_x_norm[None, :, None]).to(device)
            with torch.no_grad():
                hist_pred = model(hist_x_t)[0].cpu().numpy() * s + m
            actual_idx = start_idx + sl
            if actual_idx < len(vals):
                raw_residuals.append(vals[actual_idx] - hist_pred[0])

        if len(raw_residuals) > 5:
            clean_res = _clean_residuals(np.array(raw_residuals))
            y_pred_den = y_pred_den + _generate_future_residuals(clean_res, pl, residual_strength)

    latest_ts = sd.index.max()
    try:
        if str(freq).lower().endswith("min"):
            minutes = int(str(freq).lower()[:-3])
            start = pd.Timestamp(latest_ts) + pd.Timedelta(minutes=minutes)
        else:
            start = pd.Timestamp(latest_ts)
            start = pd.date_range(start=start, periods=2, freq=freq)[1]
        future_times = pd.date_range(start=start, periods=pl, freq=freq)
    except Exception:
        future_times = [pd.Timestamp(latest_ts)] * pl
    return build_api_response_informer(equipment_code, meas_code, freq, list(future_times), list(y_pred_den), df)


def predict_informer_api_default():
    cfg = _postgres_config_path()
    ckpt = resolve_checkpoint_path("V-SZ-ISD-102", "CCD-Score1", "informer")
    return predict_informer_api("V-SZ-ISD-102", "CCD-Score1", "15min", 365, str(ckpt), cfg)


def predict_autoformer_api(
    equipment_code: str,
    meas_code: str,
    freq: str,
    days_back: int,
    ckpt_path: str,
    config_path: str,
    add_residual: bool = True,
    residual_strength: float = 0.8,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = torch.load(ckpt_path, map_location=device)
    m = float(c["mean"])
    s = float(c["std"])
    a = c.get("args", {})
    sl = int(a.get("seq_len", 672))
    pl = int(a.get("pred_len", 288))
    model = Autoformer(
        seq_len=sl,
        label_len=int(a.get("label_len", 288)),
        pred_len=pl,
        d_model=int(a.get("d_model", 256)),
        n_heads=int(a.get("n_heads", 8)),
        d_ff=int(a.get("d_ff", 512)),
        dropout=float(a.get("dropout", 0.1)),
        moving_avg=int(a.get("moving_avg", 25)),
        e_layers=int(a.get("e_layers", 2)),
        d_layers=int(a.get("d_layers", 2)),
    ).to(device)
    model.load_state_dict(c["model"])
    model.eval()

    df = load_postgres_timeseries(config_path, equipment_code, meas_code, days_back=days_back)
    df = normalize_columns(df)
    df = parse_time(df)
    g = df[df["meas_code"] == meas_code] if "meas_code" in df.columns else df
    sd = resample_group(g, freq=freq)
    vals = pd.to_numeric(sd["value"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    if len(vals) < sl:
        return build_api_response_autoformer(equipment_code, meas_code, freq, [], [], df)
    x = vals[-sl:]
    x_norm = (x - m) / s
    x_t = torch.from_numpy(np.asarray(x_norm)[None, :, None]).to(device)

    with torch.no_grad():
        y_hat = model(x_t)
    y_pred_den = y_hat[0].detach().cpu().numpy() * s + m

    if add_residual and len(vals) > sl:
        hist_len = min(len(vals) - sl, sl * 2)
        raw_residuals = []
        for i in range(hist_len):
            start_idx = len(vals) - sl - hist_len + i
            if start_idx < 0:
                continue
            hist_x = vals[start_idx : start_idx + sl]
            hist_x_norm = (hist_x - m) / s
            hist_x_t = torch.from_numpy(hist_x_norm[None, :, None]).to(device)
            with torch.no_grad():
                hist_pred = model(hist_x_t)[0].cpu().numpy() * s + m
            actual_idx = start_idx + sl
            if actual_idx < len(vals):
                raw_residuals.append(vals[actual_idx] - hist_pred[0])

        if len(raw_residuals) > 5:
            clean_res = _clean_residuals(np.array(raw_residuals))
            y_pred_den = y_pred_den + _generate_future_residuals(clean_res, pl, residual_strength)

    latest_ts = sd.index.max()
    try:
        if str(freq).lower().endswith("min"):
            minutes = int(str(freq).lower()[:-3])
            start = pd.Timestamp(latest_ts) + pd.Timedelta(minutes=minutes)
        else:
            start = pd.Timestamp(latest_ts)
            start = pd.date_range(start=start, periods=2, freq=freq)[1]
        future_times = pd.date_range(start=start, periods=pl, freq=freq)
    except Exception:
        future_times = [pd.Timestamp(latest_ts)] * pl
    return build_api_response_autoformer(equipment_code, meas_code, freq, list(future_times), list(y_pred_den), df)

