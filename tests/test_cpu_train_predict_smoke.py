from __future__ import annotations

import json
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
    guard_report_path = root / "child-guard-report.json"
    network_attempt_path = root / "child-network-attempts.jsonl"
    guard_dir = Path(__file__).parent / "support" / "child_guard"
    child_home = root / "home"
    child_cache = root / "cache"
    child_home.mkdir()
    child_cache.mkdir()
    monkeypatch.setenv("PDM_TEST_INHERITED_SECRET", "must-not-reach-child")
    environment = {
        "CUDA_VISIBLE_DEVICES": "",
        "HOME": str(child_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(child_cache / "matplotlib"),
        "PATH": os.pathsep.join((str(console.parent), "/usr/bin", "/bin")),
        "PDM_CHILD_GUARD_REPORT": str(guard_report_path),
        "PDM_CHILD_NETWORK_SENTINEL": str(network_attempt_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": str(guard_dir),
        "VALEO_PDM_ROOT": str(root),
        "XDG_CACHE_HOME": str(child_cache),
    }
    if library_path := os.environ.get("LD_LIBRARY_PATH"):
        environment["LD_LIBRARY_PATH"] = library_path
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

    assert guard_report_path.is_file()
    guard_report = json.loads(guard_report_path.read_text(encoding="utf-8"))
    assert guard_report["guard_loaded"] is True
    assert Path(guard_report["guard_file"]).resolve() == (guard_dir / "sitecustomize.py").resolve()
    assert Path(guard_report["executable"]).resolve() == Path(sys.executable).resolve()
    assert guard_report["pid"] != os.getpid()
    assert guard_report["cuda_visible_devices"] == ""
    assert guard_report["torch_version_cuda"] is None
    assert guard_report["cuda_available"] is False
    assert guard_report["blocked_connectors"] == [
        "psycopg2.connect",
        "pyodbc.connect",
        "socket.create_connection",
        "socket.socket.connect",
        "socket.socket.connect_ex",
    ]
    assert "PDM_TEST_INHERITED_SECRET" not in guard_report["environment_keys"]
    assert not network_attempt_path.exists(), (
        network_attempt_path.read_text(encoding="utf-8") if network_attempt_path.exists() else ""
    )

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
