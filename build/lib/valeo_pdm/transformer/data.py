from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def load_series(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    col_map = {c.lower(): c for c in df.columns}
    value_key = None
    for k in ("value", "values"):
        if k in col_map:
            value_key = col_map[k]
            break
    if not value_key:
        raise ValueError(f"CSV缺少'value'列: {path}")
    df = df.rename(columns={value_key: "value"})
    if "collect_time" not in df.columns and "collecttime" in col_map:
        df = df.rename(columns={col_map["collecttime"]: "collect_time"})
    if "collect_time" in df.columns:
        df["collect_time"] = pd.to_datetime(df["collect_time"], errors="coerce")
        df = df.dropna(subset=["collect_time"])
        df = df.sort_values("collect_time")
    values = pd.to_numeric(df["value"], errors="coerce").dropna().to_numpy(dtype=np.float32)
    return values


def make_windows(values: np.ndarray, seq_len: int, label_len: int, pred_len: int, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    total = seq_len + pred_len
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for i in range(0, len(values) - total + 1, max(1, int(step))):
        window = values[i : i + total]
        x = window[:seq_len]
        y = window[-pred_len:]
        xs.append(x[:, None])
        ys.append(y)
    if not xs:
        return np.empty((0, seq_len, 1), dtype=np.float32), np.empty((0, pred_len), dtype=np.float32)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def split_train_val_test(n: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    idx = np.arange(n)
    return idx[:train_n], idx[train_n : train_n + val_n], idx[train_n + val_n :]


def normalize_train_mean_std(train_x: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = float(train_x.mean())
    std = float(train_x.std() + 1e-8)
    x_norm = (x - mean) / std
    return x_norm, mean, std

