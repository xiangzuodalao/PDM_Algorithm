from __future__ import annotations

import importlib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import yaml


FIXTURES = Path(__file__).parent / "fixtures" / "prediction_v2"
NOW = datetime(2026, 7, 29, 1, 15, tzinfo=UTC)
TENANT = "00000000-0000-4000-8000-000000000001"
ARTIFACT = (
    b'{"kind":"repeat-last","model_info_id":"pilot-fixture-v1-cnc-vibration",'
    b'"model_profile_id":"pilot-cnc-vibration","preprocessing_version":"pdm-v2-pilot-1",'
    b'"schema_version":1}'
)


def require_module(name: str, behaviour: str):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        target_or_parent = {
            ".".join(name.split(".")[:index]) for index in range(1, len(name.split(".")) + 1)
        }
        if exc.name not in target_or_parent:
            raise
        assert False, f"{behaviour} is unavailable: {name} has not been implemented"


def service_request():
    models = require_module("valeo_pdm.prediction_v2.models", "v2 strict models")
    payload = json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))
    return models.PredictionRequestV2.model_validate(payload)


def catalog(tmp_path: Path):
    module = require_module("valeo_pdm.prediction_v2.catalog", "v2 catalog-resolution")
    (tmp_path / "fixture.json").write_bytes(ARTIFACT)
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "entries": [
                    {
                        "model_profile_id": "pilot-cnc-vibration",
                        "model_info_id": "pilot-fixture-v1-cnc-vibration",
                        "meas_code": "vibration_rms",
                        "unit": "mm/s",
                        "value_scale": 2,
                        "sampling_frequency": "1min",
                        "request_window_points": 66,
                        "context_points": 60,
                        "horizon_points": 15,
                        "preprocessing_version": "pdm-v2-pilot-1",
                        "artifact_path": "fixture.json",
                        "artifact_sha256": hashlib.sha256(ARTIFACT).hexdigest(),
                    }
                ]
            }
        )
    )
    return module.ModelCatalog.from_manifest(
        manifest, object_root=tmp_path, allowed_tenant_ids={TENANT}
    )


def test_repeat_last_forecast_is_deterministic_and_exact_horizon(tmp_path: Path) -> None:
    """Replacing repeat-last or its one-minute timing would change the deterministic pilot forecast."""
    service_module = require_module("valeo_pdm.prediction_v2.service", "v2 repeat-last-forecast")
    response = service_module.PredictionV2Service(catalog(tmp_path)).predict(
        service_request(), now=NOW
    )
    assert len(response.forecast) == 15
    assert [(point.timestamp, point.value, point.unit) for point in response.forecast[:2]] == [
        (1785287700000, "4.00", "mm/s"),
        (1785287760000, "4.00", "mm/s"),
    ]
    assert (
        response.input_digest == "c67fe58c14f2e8a1ef252e6ff47b4877c0da8fa52664772dd3c08ae3d6552609"
    )
    assert (
        response.request_digest
        == "067096fc115546164d8608cddadc56b185fefea9dffab4c7462c8d28011cdfbd"
    )
    assert (
        response.model_artifact_sha256
        == "5feeb31058fe0521f94758faa22214afe4619e466e22cbb8c6fb7ffcf2369562"
    )
    assert response.generated_at == NOW
    assert response.forecast[-1].timestamp == 1785288540000


def test_repeat_last_changes_only_with_last_valid_value_and_never_loads_model_stacks(
    tmp_path: Path, monkeypatch
) -> None:
    """A model/DB/CSV access or wrong final sample would violate the fixture-only v2 boundary."""
    service_module = require_module("valeo_pdm.prediction_v2.service", "v2 repeat-last-forecast")
    original = service_request()
    changed_payload = original.model_dump(mode="json")
    changed_payload["history"][-1]["value"] = "7.00"
    models = require_module("valeo_pdm.prediction_v2.models", "v2 strict models")
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 request digest normalization"
    )
    draft = models.PredictionRequestV2.model_validate(changed_payload)
    changed = models.PredictionRequestV2.model_validate(
        {**changed_payload, "request_digest": normalization.calculate_request_digest(draft)}
    )
    predict = service_module.PredictionV2Service(catalog(tmp_path)).predict
    first = predict(original, now=NOW)
    second = predict(original, now=NOW)
    different = predict(changed, now=NOW)
    assert first == second
    assert first.forecast[-1].value == "4.00"
    assert different.forecast[-1].value == "7.00"


def test_service_canonicalizes_before_digest_and_repeats_last_non_null(tmp_path: Path) -> None:
    """Verifying raw request text or using trailing null would break deterministic replay."""
    service_module = require_module("valeo_pdm.prediction_v2.service", "v2 repeat-last-forecast")
    models = require_module("valeo_pdm.prediction_v2.models", "v2 strict models")
    original = service_request()
    raw = original.model_dump(mode="json")
    raw["unit"] = " mm/s "
    raw["history"] = [
        {**point, "unit": " mm/s ", "value": "4.000"} for point in raw["history"][:-2]
    ]
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 canonical request normalization"
    )
    draft = models.PredictionRequestV2.model_validate(raw)
    model_catalog = catalog(tmp_path)
    model_profile = model_catalog.resolve(
        TENANT, "pilot-cnc-vibration", "pilot-fixture-v1-cnc-vibration", "vibration_rms"
    ).profile
    raw["request_digest"] = normalization.calculate_request_digest(
        normalization.canonicalize_request(draft, model_profile)
    )
    response = service_module.PredictionV2Service(model_catalog).predict(
        models.PredictionRequestV2.model_validate(raw), now=NOW
    )
    assert response.forecast[0].value == "4.00"
    assert (
        response.input_digest != "c67fe58c14f2e8a1ef252e6ff47b4877c0da8fa52664772dd3c08ae3d6552609"
    )
