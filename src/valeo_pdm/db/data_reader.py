from __future__ import annotations

import json
from datetime import datetime, timedelta
import pandas as pd
import pyodbc


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


def load_postgres_timeseries(
    config_path: str, equipment_code: str, meas_code: str, days_back: int = 30
):
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


def load_sqlserver_timeseries(
    config_path: str, equipment_code: str, meas_code: str, days_back: int
) -> pd.DataFrame:
    """
    从 SQL Server 数据库中拉取指定设备和测点的时序数据。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    conn_str = cfg.get("conn_str")
    if not conn_str:
        raise ValueError("SQL Server 配置文件中缺少 conn_str 参数")

    # 1. 和 postgres 保持完全一致的时间格式与截止时间计算
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d %H:%M:%S")
    end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 2. 增加 end_date 条件，保持数据截取窗口严谨一致
    # 如果您的 json 配置文件中已经有了 sql_query，这里也可以改成 cfg.get("sql_query", """...""")
    query = """
        SELECT 
            f_id as id, EquipmentCode as equipment_code, MeasCode as meas_code, MonitorValue as value, Unit as unit, CollectTime as collect_time
        FROM mom_bas_ai_model_train_data 
        WHERE EquipmentCode = ? 
          AND MeasCode = ? 
          AND CollectTime >= ?
          AND CollectTime <= ?
          AND IsActive is null
        ORDER BY CollectTime ASC
    """

    print("正在连接 SQL Server 读取时序数据")
    print(f"正在从 SQL Server 拉取 {equipment_code} - {meas_code} 过去 {days_back} 天的数据...")

    with pyodbc.connect(conn_str) as conn:
        with conn.cursor() as cursor:
            # 3. 采用游标执行，传入 4 个参数
            cursor.execute(query, (equipment_code, meas_code, start_date, end_date))

            # 4. 获取列名，和 Postgres 逻辑完全一致
            cols = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()

            # 5. 组装 DataFrame
            # 注意：pyodbc 返回的是 Row 对象，显式转为 tuple 能避免 Pandas 版本兼容性警告，且形态与 psycopg2 完美对齐
            df = pd.DataFrame([tuple(row) for row in rows], columns=cols)

    if df.empty:
        raise ValueError(
            f"SQL Server 数据源返回空数据! 请检查设备 {equipment_code} 在过去 {days_back} 天内是否有数据。"
        )

    return df
