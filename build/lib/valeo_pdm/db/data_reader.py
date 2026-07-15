from __future__ import annotations

import json
from datetime import datetime, timedelta


def load_sqlserver_classification(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    try:
        from fetchdb import fetch_from_db as fdb
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少依赖，请先安装 pandas 与 pyodbc：pip install pandas pyodbc"
        ) from e
    x, y = fdb(cfg["conn_str"], cfg["sql_query"], cfg["label"])
    return x, y


def _parse_pg_conn_str(s: str):
    d = {}
    for item in s.split():
        k, v = item.split("=", 1)
        d[k.strip()] = v.strip()
    return d


def load_postgres_timeseries(config_path: str, equipment_code: str, meas_code: str, days_back: int = 30):
    try:
        import pandas as pd
        import psycopg2
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "缺少依赖，请先安装 pandas 与 psycopg2-binary：pip install pandas psycopg2-binary"
        ) from e

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with psycopg2.connect(**_parse_pg_conn_str(cfg["conn_str"])) as conn:
        with conn.cursor() as cursor:
            cursor.execute(cfg["sql_query"], (equipment_code, meas_code, start_date, end_date))
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            df = pd.DataFrame(rows, columns=cols)
    return df


def preprocess_timeseries(df):
    import pandas as pd

    df = df.copy()
    if df.empty:
        return df
    df["collect_time"] = pd.to_datetime(df["collect_time"], errors="coerce")
    df = df.dropna(subset=["collect_time"])
    df["collect_time"] = df["collect_time"].dt.floor("min")
    df = df.drop_duplicates(subset="collect_time", keep="first")
    df = df.sort_values("collect_time")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])
    if df["value"].isna().sum() > 0:
        df["value"] = df["value"].interpolate(method="linear")
    return df

