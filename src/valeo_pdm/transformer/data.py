from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


TIME_COLUMN_ALIASES = ("collect_time", "collecttime", "timestamp")
VALUE_COLUMN_ALIASES = ("value", "values")


def _alias_column(columns: dict[str, str], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    return None


def normalize_timeseries_columns(df: pd.DataFrame) -> pd.DataFrame:
    """按不区分大小写的项目数据契约规范化时序列名。"""

    columns = {str(column).lower(): str(column) for column in df.columns}
    time_column = _alias_column(columns, TIME_COLUMN_ALIASES)
    value_column = _alias_column(columns, VALUE_COLUMN_ALIASES)
    if time_column is None:
        raise ValueError("时序数据缺少时间列: collect_time/collecttime/timestamp")
    if value_column is None:
        raise ValueError("时序数据缺少数值列: value/values")

    rename_map = {time_column: "collect_time", value_column: "value"}
    for canonical in ("equipment_code", "meas_code", "unit"):
        source = columns.get(canonical)
        if source is not None:
            rename_map[source] = canonical
    return df.rename(columns=rename_map).copy()


def clean_and_resample_timeseries(
    df: pd.DataFrame,
    freq: str,
    *,
    equipment_code: str | None = None,
    meas_code: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """过滤目标场景、丢弃坏行，并按时间桶均值聚合。"""

    normalized = normalize_timeseries_columns(df)
    input_rows = len(normalized)
    filtered = normalized
    if equipment_code is not None and "equipment_code" in filtered.columns:
        filtered = filtered[filtered["equipment_code"].astype(str) == str(equipment_code)]
    if meas_code is not None and "meas_code" in filtered.columns:
        filtered = filtered[filtered["meas_code"].astype(str) == str(meas_code)]
    filtered = filtered.copy()

    parsed_time = pd.to_datetime(
        filtered["collect_time"], errors="coerce", format="mixed", utc=True
    )
    parsed_value = pd.to_numeric(filtered["value"], errors="coerce")
    invalid_time_rows = int(parsed_time.isna().sum())
    invalid_value_rows = int(parsed_value.isna().sum())
    filtered["collect_time"] = parsed_time
    filtered["value"] = parsed_value
    cleaned = filtered.dropna(subset=["collect_time", "value"]).sort_values("collect_time")

    if cleaned.empty:
        result = pd.DataFrame(
            {"value": pd.Series(dtype="float64")},
            index=pd.DatetimeIndex([], name="collect_time"),
        )
    else:
        buckets = cleaned["collect_time"].dt.floor(freq)
        values = pd.DataFrame({"bucket": buckets, "value": cleaned["value"]})
        result = values.groupby("bucket", as_index=True)["value"].mean().sort_index().to_frame()
        result.index.name = "collect_time"

    report = {
        "input_rows": int(input_rows),
        "filtered_rows": int(len(filtered)),
        "dropped_invalid_time_rows": invalid_time_rows,
        "dropped_invalid_value_rows": invalid_value_rows,
        "valid_rows_before_resample": int(len(cleaned)),
        "aggregated_rows": int(max(0, len(cleaned) - len(result))),
        "rows_after_resample": int(len(result)),
    }
    return result, report


def load_timeseries_file(
    path: str | Path,
    freq: str,
    *,
    equipment_code: str | None = None,
    meas_code: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".json":
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        if not isinstance(payload, list):
            raise ValueError("JSON 时序数据必须是行数组或包含 rows")
        df = pd.DataFrame(payload)
    else:
        df = pd.read_csv(source_path)
    return clean_and_resample_timeseries(
        df,
        freq,
        equipment_code=equipment_code,
        meas_code=meas_code,
    )


def load_series(path: str, freq: str = "15min") -> np.ndarray:
    series_df, _ = load_timeseries_file(path, freq)
    return series_df["value"].to_numpy(dtype=np.float32)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} 必须是非布尔正整数")
    return value


def validate_window_params(
    seq_len: object, label_len: object, pred_len: object, stride: object = 1
) -> int:
    seq = _positive_int(seq_len, "seq_len")
    label = _positive_int(label_len, "label_len")
    _positive_int(pred_len, "pred_len")
    step = _positive_int(stride, "stride")
    if label > seq:
        raise ValueError("label_len 必须小于或等于 seq_len")
    return step


def count_windows(rows: int, seq_len: int, pred_len: int, stride: int = 1) -> int:
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("rows 必须是非负整数")
    step = _positive_int(stride, "stride")
    _positive_int(seq_len, "seq_len")
    _positive_int(pred_len, "pred_len")
    if rows < seq_len + pred_len:
        return 0
    return ((rows - seq_len - pred_len) // step) + 1


def window_split_counts(
    rows: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    stride: int = 1,
) -> dict[str, int]:
    step = validate_window_params(seq_len, label_len, pred_len, stride)
    windows = count_windows(rows, seq_len, pred_len, step)
    train_idx, val_idx, test_idx = split_train_val_test(windows)
    return {
        "rows": rows,
        "windows": windows,
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "test_windows": int(len(test_idx)),
        "stride": step,
    }


def require_usable_window_splits(
    rows: int,
    seq_len: int,
    label_len: int,
    pred_len: int,
    stride: int = 1,
) -> dict[str, int]:
    counts = window_split_counts(rows, seq_len, label_len, pred_len, stride)
    if min(counts["train_windows"], counts["val_windows"], counts["test_windows"]) < 1:
        raise ValueError(
            "清洗重采样后的窗口不足: "
            f"total={counts['windows']}, train={counts['train_windows']}, "
            f"val={counts['val_windows']}, test={counts['test_windows']}"
        )
    return counts


def minimum_rows_for_usable_window_splits(
    seq_len: int,
    label_len: int,
    pred_len: int,
    stride: int = 1,
) -> int:
    """返回当前顺序 80/10/10 切分下三段均有窗口所需的最少清洗行数。"""

    step = validate_window_params(seq_len, label_len, pred_len, stride)
    window_count = 1
    while True:
        train_idx, val_idx, test_idx = split_train_val_test(window_count)
        if min(len(train_idx), len(val_idx), len(test_idx)) >= 1:
            return seq_len + pred_len + (window_count - 1) * step
        window_count += 1


def make_windows(
    values: np.ndarray, seq_len: int, label_len: int, pred_len: int, step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    step = validate_window_params(seq_len, label_len, pred_len, step)
    total = seq_len + pred_len
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(0, len(values) - total + 1, step):
        window = values[i : i + total]
        x = window[:seq_len]
        y = window[-pred_len:]
        xs.append(x[:, None])
        ys.append(y)
    if not xs:
        return np.empty((0, seq_len, 1), dtype=np.float32), np.empty(
            (0, pred_len), dtype=np.float32
        )
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def split_train_val_test(
    n: int, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n 必须是非负整数")
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    idx = np.arange(n)
    return idx[:train_n], idx[train_n : train_n + val_n], idx[train_n + val_n :]


def normalize_train_mean_std(train_x: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    if len(train_x) == 0:
        raise ValueError("训练窗口为空，无法计算归一化参数")
    mean = float(train_x.mean())
    std = float(train_x.std() + 1e-8)
    x_norm = (x - mean) / std
    return x_norm, mean, std
