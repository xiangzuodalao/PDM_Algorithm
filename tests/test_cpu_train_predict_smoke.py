from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from valeo_pdm.api.app import app


def test_installed_console_trains_and_predicts_with_real_informer_on_cpu(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "cpu-smoke"
    configs = root / "configs"
    data = root / "data"
    configs.mkdir(parents=True)
    data.mkdir()

    rows = ["collect_time,value"]
    rows.extend(f"2026-01-01 {hour:02d}:00:00,{20 + hour / 10:.1f}" for hour in range(20))
    (data / "cpu-smoke.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (configs / "model_registry.yaml").write_text(
        """
models:
  CPU-SMOKE:
    TEMP:
      model_type: informer
      freq: "1h"
      days_back: 1
      source: csv
      data_path: data/cpu-smoke.csv
      train_params:
        seq_len: 4
        label_len: 2
        pred_len: 2
        batch_size: 8
        epochs: 1
        lr: 0.0005
        d_model: 8
        n_heads: 1
        d_ff: 8
        dropout: 0
        e_layers: 1
        d_layers: 1
        attn_type: full
        distil: false
""".lstrip(),
        encoding="utf-8",
    )

    console = Path(sys.executable).with_name("valeo-pdm")
    environment = os.environ.copy()
    environment["VALEO_PDM_ROOT"] = str(root)
    environment["MPLBACKEND"] = "Agg"
    environment.pop("VALEO_PDM_MODEL_REGISTRY", None)
    environment.pop("VALEO_PDM_POSTGRES_CONFIG", None)
    environment.pop("VALEO_PDM_SQLSERVER_CONFIG", None)
    result = subprocess.run(
        [str(console), "train", "-e", "CPU-SMOKE", "-m", "TEMP"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"

    checkpoint = (
        root / "artifacts" / "checkpoints" / "exp_CPU-SMOKE_TEMP_informer" / "informer_best.pt"
    )
    assert checkpoint.is_file()

    monkeypatch.setenv("VALEO_PDM_ROOT", str(root))
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.delenv("VALEO_PDM_MODEL_REGISTRY", raising=False)
    monkeypatch.delenv("VALEO_PDM_POSTGRES_CONFIG", raising=False)
    monkeypatch.delenv("VALEO_PDM_SQLSERVER_CONFIG", raising=False)
    with TestClient(app) as client:
        response = client.post(
            "/measPredict/predict",
            json={
                "EquipmentCode": "CPU-SMOKE",
                "MeasCode": "TEMP",
                "ModelType": "informer",
                "HistoryData": [],
            },
        )

    assert response.status_code == 200, response.text
    values = response.json()["response"]["Values"]
    assert len(values) == 2
    assert all(math.isfinite(float(item["Value"])) for item in values)
