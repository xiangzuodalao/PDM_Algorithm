from __future__ import annotations

import argparse
import os
from pathlib import Path

from valeo_pdm.paths import data_dir
from valeo_pdm.transformer.data import load_timeseries_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resample time-series CSV to fixed frequency")
    parser.add_argument("--in", dest="in_path", required=True, help="输入CSV路径")
    parser.add_argument(
        "--out", dest="out_path", default="", help="输出CSV路径（默认 data/processed/<name>.csv）"
    )
    parser.add_argument(
        "--freq", dest="freq", default="15min", help="重采样频率，例如 15min/2min/1h"
    )
    args = parser.parse_args(argv)

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {in_path}")
    out, _quality = load_timeseries_file(in_path, args.freq)
    out = out.reset_index()

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
