from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "onboard_scenario"
SCRIPT = ROOT / ".agents" / "skills" / "pdm-onboard-scenario" / "scripts" / "onboard_scenario.py"
SPEC = importlib.util.spec_from_file_location("pdm_onboard_scenario_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ONBOARD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ONBOARD
SPEC.loader.exec_module(ONBOARD)


@pytest.fixture
def registry_copy() -> Path:
    destination = ROOT / "tests" / f".registry-{uuid4().hex}.yaml"
    shutil.copyfile(FIXTURES / "registry.yaml", destination)
    try:
        yield destination
    finally:
        destination.unlink(missing_ok=True)


def _run(candidate: Path, registry: Path, **overrides: object) -> dict[str, object]:
    params: dict[str, object] = {
        "repo_root": ROOT,
        "candidate_path": candidate,
        "db_models_path": FIXTURES / "db_models.json",
        "registry_path": registry,
    }
    params.update(overrides)
    return ONBOARD.onboard_scenario(**params)


def _candidate(tmp_path: Path, equipment: str, meas: str, config: dict[str, object]) -> Path:
    path = tmp_path / "candidate.yaml"
    path.write_text(
        yaml.safe_dump(
            {"equipment_code": equipment, "meas_code": meas, "config": config},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _valid_config(**updates: object) -> dict[str, object]:
    config: dict[str, object] = {
        "model_type": "informer",
        "freq": "1h",
        "days_back": 7,
        "source": "db",
        "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2},
    }
    config.update(updates)
    return config


def test_db_fixture_preview_and_apply_preserve_unrelated_text(registry_copy: Path) -> None:
    original = registry_copy.read_text(encoding="utf-8")
    kwargs = {"timeseries_fixture": FIXTURES / "db_timeseries.csv"}

    preview = _run(FIXTURES / "candidate_db.yaml", registry_copy, **kwargs)

    assert preview["status"] == "preview"
    assert preview["can_apply"] is True
    assert preview["effective_config"] == {
        "model_type": "informer",
        "freq": "1h",
        "days_back": 7,
        "source": "db",
        "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2},
    }
    assert preview["window_counts"] == {
        "rows": 20,
        "windows": 15,
        "train_windows": 12,
        "val_windows": 1,
        "test_windows": 2,
        "stride": 1,
    }
    assert "KEEP-EQ" not in preview["diff"]
    assert str(registry_copy.parent) not in preview["diff"]
    assert registry_copy.read_text(encoding="utf-8") == original

    applied = _run(
        FIXTURES / "candidate_db.yaml",
        registry_copy,
        apply=True,
        confirmed=True,
        expected_diff_hash=preview["diff_hash"],
        **kwargs,
    )

    assert applied["status"] == "applied"
    updated = registry_copy.read_text(encoding="utf-8")
    assert "# Test registry: unrelated comments and fields must survive onboarding." in updated
    assert "keep_marker: untouched" in updated
    parsed = yaml.safe_load(updated)
    assert parsed["models"]["KEEP-EQ"]["KEEP-MEAS"]["keep_marker"] == "untouched"
    assert parsed["models"]["DB-NEW"]["TEMP"] == {
        "train_params": {"seq_len": 4, "label_len": 2, "pred_len": 2}
    }


def test_csv_fixture_preview_supports_case_insensitive_columns(registry_copy: Path) -> None:
    result = _run(FIXTURES / "candidate_csv.yaml", registry_copy)

    assert result["effective_config"]["model_type"] == "autoformer"
    assert result["effective_config"]["days_back"] == 7
    assert result["model_support"]["supported"] == ["autoformer", "informer"]
    assert result["data_quality"]["dropped_invalid_time_rows"] == 1
    assert result["data_quality"]["dropped_invalid_value_rows"] == 1
    assert result["data_quality"]["aggregated_rows"] == 1
    assert result["data_quality"]["rows_after_resample"] == 20


@pytest.mark.parametrize(
    ("equipment", "meas", "config", "error_code"),
    [
        ("KEEP-EQ", "KEEP-MEAS", _valid_config(), "E_DUPLICATE_YAML"),
        ("DB-EXIST", "MEAS-EXIST", _valid_config(), "E_DUPLICATE_DB"),
        ("NEW", "UNKNOWN", _valid_config(model_type="xlstm"), "E_MODEL_UNSUPPORTED"),
        (
            "NEW",
            "BOOL",
            _valid_config(train_params={"seq_len": True, "label_len": 1, "pred_len": 1}),
            "E_WINDOW_INVALID",
        ),
        (
            "NEW",
            "LABEL",
            _valid_config(train_params={"seq_len": 2, "label_len": 3, "pred_len": 1}),
            "E_WINDOW_INVALID",
        ),
        (
            "NEW",
            "MISSING",
            _valid_config(source="csv", data_path="tests/fixtures/onboard_scenario/missing.csv"),
            "E_DATA_FILE_MISSING",
        ),
        (
            "NEW",
            "NO-PATH",
            _valid_config(source="csv"),
            "E_DATA_PATH_REQUIRED",
        ),
        (
            "NEW",
            "BAD-SOURCE",
            _valid_config(source="ftp"),
            "E_SOURCE_UNSUPPORTED",
        ),
        (
            "NEW",
            "BAD-STRIDE",
            _valid_config(train_params={"seq_len": 4, "label_len": 2, "pred_len": 2, "stride": 0}),
            "E_WINDOW_INVALID",
        ),
    ],
)
def test_stable_errors(
    tmp_path: Path,
    registry_copy: Path,
    equipment: str,
    meas: str,
    config: dict[str, object],
    error_code: str,
) -> None:
    candidate = _candidate(tmp_path, equipment, meas, config)

    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            candidate,
            registry_copy,
            timeseries_fixture=FIXTURES / "db_timeseries.csv",
        )

    assert caught.value.code == error_code


@pytest.mark.parametrize(
    "fixture_name",
    ["db_timeseries_missing_unit.csv", "db_timeseries_wrong_target.csv"],
)
def test_db_fixture_requires_full_schema_and_target_rows(
    registry_copy: Path, fixture_name: str
) -> None:
    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            FIXTURES / "candidate_db.yaml",
            registry_copy,
            timeseries_fixture=FIXTURES / fixture_name,
        )

    assert caught.value.code == "E_DATA_SCHEMA"


def test_db_models_fixture_cannot_masquerade_as_verified_config(
    tmp_path: Path, registry_copy: Path
) -> None:
    invalid = tmp_path / "db_models.json"
    invalid.write_text('{"conn_str": "must-not-be-read"}', encoding="utf-8")

    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            FIXTURES / "candidate_db.yaml",
            registry_copy,
            db_models_path=invalid,
            timeseries_fixture=FIXTURES / "db_timeseries.csv",
        )

    assert caught.value.code == "E_CANDIDATE_SCHEMA"
    assert "must-not-be-read" not in caught.value.message


def test_csv_requires_time_and_value_columns(tmp_path: Path, registry_copy: Path) -> None:
    candidate = _candidate(
        tmp_path,
        "CSV-BAD",
        "TEMP",
        _valid_config(
            source="csv",
            data_path="tests/fixtures/onboard_scenario/csv_missing_time.csv",
        ),
    )

    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(candidate, registry_copy)

    assert caught.value.code == "E_DATA_SCHEMA"


def test_candidate_values_are_normalized_before_diff_and_write(
    tmp_path: Path, registry_copy: Path
) -> None:
    candidate = _candidate(
        tmp_path,
        "CSV-NORMALIZED",
        "TEMP",
        _valid_config(
            model_type=" AUTOFORMER ",
            source=" CSV ",
            freq=" 1h ",
            data_path=" tests/fixtures/onboard_scenario/csv_timeseries.csv ",
        ),
    )

    result = _run(candidate, registry_copy)

    assert result["effective_config"]["model_type"] == "autoformer"
    assert result["effective_config"]["source"] == "csv"
    assert result["effective_config"]["freq"] == "1h"
    assert "AUTOFORMER" not in result["diff"]
    assert "source: csv" in result["diff"]


def test_external_csv_path_requires_explicit_authorization(
    tmp_path: Path, registry_copy: Path
) -> None:
    external = tmp_path / "external.csv"
    external.write_text("collect_time,value\n2026-01-01,1\n", encoding="utf-8")
    candidate = _candidate(
        tmp_path,
        "CSV-EXTERNAL",
        "TEMP",
        _valid_config(source="csv", data_path=str(external)),
    )

    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(candidate, registry_copy)

    assert caught.value.code == "E_EXTERNAL_DATA_PATH"
    assert str(tmp_path) not in str(caught.value.details)


def test_db_unknown_allows_preview_but_blocks_apply(registry_copy: Path) -> None:
    preview = _run(
        FIXTURES / "candidate_db.yaml",
        registry_copy,
        db_models_path=None,
        timeseries_fixture=FIXTURES / "db_timeseries.csv",
    )
    assert preview["can_apply"] is False
    assert preview["db_status"] == "unknown"

    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            FIXTURES / "candidate_db.yaml",
            registry_copy,
            db_models_path=None,
            timeseries_fixture=FIXTURES / "db_timeseries.csv",
            apply=True,
            confirmed=True,
        )
    assert caught.value.code == "E_DB_STATUS_UNKNOWN"


def test_apply_requires_explicit_confirmation(registry_copy: Path) -> None:
    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            FIXTURES / "candidate_db.yaml",
            registry_copy,
            timeseries_fixture=FIXTURES / "db_timeseries.csv",
            apply=True,
        )
    assert caught.value.code == "E_CONFIRM_REQUIRED"


def test_apply_requires_the_confirmed_preview_hash(registry_copy: Path) -> None:
    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            FIXTURES / "candidate_db.yaml",
            registry_copy,
            timeseries_fixture=FIXTURES / "db_timeseries.csv",
            apply=True,
            confirmed=True,
            expected_diff_hash="different",
        )
    assert caught.value.code == "E_DIFF_MISMATCH"


def test_confirmation_hash_binds_registry_defaults_snapshot(registry_copy: Path) -> None:
    preview = _run(
        FIXTURES / "candidate_db.yaml",
        registry_copy,
        timeseries_fixture=FIXTURES / "db_timeseries.csv",
    )
    changed = registry_copy.read_text(encoding="utf-8").replace("days_back: 7", "days_back: 8")
    registry_copy.write_text(changed, encoding="utf-8")

    with pytest.raises(ONBOARD.OnboardError) as caught:
        _run(
            FIXTURES / "candidate_db.yaml",
            registry_copy,
            timeseries_fixture=FIXTURES / "db_timeseries.csv",
            apply=True,
            confirmed=True,
            expected_diff_hash=preview["diff_hash"],
        )

    assert caught.value.code == "E_DIFF_MISMATCH"
    assert "DB-NEW" not in registry_copy.read_text(encoding="utf-8")


def test_post_write_failure_restores_original(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("models: {}\n", encoding="utf-8")
    calls = 0

    def validator(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated post-write failure")

    with pytest.raises(ONBOARD.OnboardError) as caught:
        ONBOARD.atomic_replace_with_rollback(registry, "models:\n  NEW: {}\n", validator)

    assert caught.value.code == "E_POST_WRITE_VALIDATION"
    assert registry.read_text(encoding="utf-8") == "models: {}\n"


def test_apply_rejects_registry_changed_since_preview(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("models: {}\n", encoding="utf-8")

    with pytest.raises(ONBOARD.OnboardError) as caught:
        ONBOARD.atomic_replace_with_rollback(
            registry,
            "models:\n  NEW: {}\n",
            lambda _path: None,
            expected_original=b"models:\n  DIFFERENT: {}\n",
        )

    assert caught.value.code == "E_REGISTRY_CHANGED"
    assert registry.read_text(encoding="utf-8") == "models: {}\n"


def test_apply_rejects_existing_onboard_lock(tmp_path: Path) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text("models: {}\n", encoding="utf-8")
    lock = registry.with_name(f".{registry.name}.onboard.lock")
    lock.write_text("", encoding="utf-8")

    with pytest.raises(ONBOARD.OnboardError) as caught:
        ONBOARD.atomic_replace_with_rollback(
            registry,
            "models:\n  NEW: {}\n",
            lambda _path: None,
            expected_original=registry.read_bytes(),
        )

    assert caught.value.code == "E_REGISTRY_CHANGED"
    assert registry.read_text(encoding="utf-8") == "models: {}\n"
    assert lock.exists()
