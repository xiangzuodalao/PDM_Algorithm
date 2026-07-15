from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from valeo_pdm.paths import data_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resample time-series CSV to fixed frequency")
    parser.add_argument("--in", dest="in_path", required=True, help="输入CSV路径")
    parser.add_argument("--out", dest="out_path", default="", help="输出CSV路径（默认 data/processed/<name>.csv）")
    parser.add_argument("--freq", dest="freq", default="15min", help="重采样频率，例如 15min/2min/1h")
    args = parser.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")
    df = pd.read_csv(in_path)
    cols = {c.lower(): c for c in df.columns}
    if "collect_time" in cols:
        df = df.rename(columns={cols["collect_time"]: "collect_time"})
    elif "timestamp" in cols:
        df = df.rename(columns={cols["timestamp"]: "collect_time"})
    if "value" in cols:
        df = df.rename(columns={cols["value"]: "value"})
    df["collect_time"] = pd.to_datetime(df["collect_time"], errors="coerce")
    df = df.dropna(subset=["collect_time"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).sort_values("collect_time")
    df = df.set_index("collect_time")
    out = df.resample(args.freq).mean().dropna().reset_index()

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        out_dir = data_dir() / "processed"
        os.makedirs(out_dir, exist_ok=True)
        out_path = out_dir / in_path.name
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
