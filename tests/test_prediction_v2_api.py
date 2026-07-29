from __future__ import annotations

import hashlib
import json
import socket
import sys
import types
from pathlib import Path

import yaml
import rfc8785
import pytest
from fastapi.testclient import TestClient


# The production image supplies unixODBC; this isolated HTTP boundary test does not.
sys.modules.setdefault("pyodbc", types.SimpleNamespace(connect=lambda *_args, **_kwargs: None))


FIXTURES = Path(__file__).parent / "fixtures" / "prediction_v2"
TOKEN = "opaque-pdm-v2-token"
ARTIFACT = (
    b'{"kind":"repeat-last","model_info_id":"pilot-fixture-v1-cnc-vibration",'
    b'"model_profile_id":"pilot-cnc-vibration","preprocessing_version":"pdm-v2-pilot-1",'
    b'"schema_version":1}'
)


def payload() -> dict:
    return json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))


def request_digest(body: dict) -> str:
    projection = {
        key: body[key]
        for key in (
            "tenant_id",
            "equipment_id",
            "model_profile_id",
            "model_info_id",
            "meas_code",
            "unit",
            "sampling_frequency",
            "window_start",
            "window_end",
        )
    }
    projection["history"] = sorted(
        body["history"], key=lambda point: (point["timestamp"], point["data_id"])
    )
    return hashlib.sha256(rfc8785.dumps(projection)).hexdigest()


def configure_runtime(monkeypatch, tmp_path: Path) -> None:
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
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN", TOKEN)
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_MANIFEST", str(manifest))
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_OBJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("VALEO_PDM_ALLOWED_TENANT_IDS", "00000000-0000-4000-8000-000000000001")


def post(client: TestClient, body: dict, token: str = TOKEN):
    return client.post(
        "/api/v2/predictions", json=body, headers={"Authorization": f"Bearer {token}"}
    )


def test_prediction_v2_valid_contract_request_returns_exact_forecast(
    monkeypatch, tmp_path: Path
) -> None:
    """Removing the v2 provider route would turn a valid contract request into a 404."""
    configure_runtime(monkeypatch, tmp_path)
    from valeo_pdm.api import router
    from valeo_pdm.api.app import app

    monkeypatch.setattr(
        router,
        "get_model_config",
        lambda *_args: (_ for _ in ()).throw(AssertionError("legacy config accessed")),
    )
    for name in (
        "get_trainer",
        "load_timeseries_file",
        "load_postgres_timeseries",
        "load_sqlserver_timeseries",
    ):
        monkeypatch.setattr(
            router,
            name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy access")),
        )
    monkeypatch.setattr(
        socket,
        "create_connection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network access")),
    )
    original_read_bytes = Path.read_bytes

    def reject_checkpoint(path: Path) -> bytes:
        if path.suffix == ".pt":
            raise AssertionError("checkpoint access")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_checkpoint)

    with TestClient(app) as client:
        response = post(client, payload())
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {
        "correlation_id",
        "equipment_id",
        "model_profile_id",
        "model_info_id",
        "model_artifact_sha256",
        "meas_code",
        "request_digest",
        "input_digest",
        "generated_at",
        "forecast",
    }
    assert len(body["forecast"]) == 15
    assert "risk" not in body and "recommendation" not in body


def test_v2_ignores_legacy_allowlist_environment_name(monkeypatch, tmp_path: Path) -> None:
    """The retired allowlist variable must not accidentally authorize a tenant."""
    configure_runtime(monkeypatch, tmp_path)
    monkeypatch.delenv("VALEO_PDM_ALLOWED_TENANT_IDS")
    monkeypatch.setenv(
        "VALEO_PDM_PREDICTION_V2_ALLOWED_TENANT_IDS", "00000000-0000-4000-8000-000000000001"
    )
    from valeo_pdm.api.app import app

    with TestClient(app) as client:
        response = post(client, payload())
    assert response.status_code == 503


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "00000000-0000-4000-8000-00000000000A"),
        ("tenant_id", "00000000000040008000000000000001"),
        ("tenant_id", "{00000000-0000-4000-8000-000000000001}"),
        ("tenant_id", 1),
        ("tenant_id", "not-a-uuid"),
        ("window_start", "1785283740000"),
        ("window_start", 1785283740000.0),
        ("window_start", True),
    ],
)
def test_v2_http_boundary_rejects_noncanonical_uuid_and_noninteger_tokens(
    monkeypatch, tmp_path: Path, field: str, value: object
) -> None:
    """Relaxed HTTP coercion would admit payloads that the provider contract forbids."""
    configure_runtime(monkeypatch, tmp_path)
    from valeo_pdm.api.app import app

    body = payload()
    body[field] = value
    with TestClient(app) as client:
        response = post(client, body)
    assert response.status_code == 422
    assert response.json() == {
        "code": "INVALID_PREDICTION_REQUEST",
        "message": "Prediction request is invalid.",
    }


def test_v2_authentication_precedes_catalog_and_has_one_safe_401(monkeypatch) -> None:
    """Resolving a catalog before opaque-token authentication could disclose tenant model existence."""
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN", TOKEN)
    monkeypatch.delenv("VALEO_PDM_PREDICTION_V2_MANIFEST", raising=False)
    from valeo_pdm.api.app import app

    with TestClient(app) as client:
        missing = client.post("/api/v2/predictions", json=payload())
        malformed = client.post(
            "/api/v2/predictions", json=payload(), headers={"Authorization": "Basic bad"}
        )
        wrong = post(client, payload(), token="wrong")
    assert missing.status_code == malformed.status_code == wrong.status_code == 401
    assert (
        missing.json()
        == malformed.json()
        == wrong.json()
        == {"code": "PREDICTION_UNAUTHORIZED", "message": "Authentication failed."}
    )


def test_v2_validation_digest_and_unknown_model_use_stable_root_errors(
    monkeypatch, tmp_path: Path
) -> None:
    """Default framework validation or unsafely mapped domain errors would violate the provider contract."""
    configure_runtime(monkeypatch, tmp_path)
    from valeo_pdm.api.app import app

    bad_decimal = payload()
    bad_decimal["history"][0]["value"] = "1e2"
    bad_decimal["request_digest"] = request_digest(bad_decimal)
    bad_digest = {**payload(), "request_digest": "0" * 64}
    unknown = {**payload(), "model_info_id": "not-present"}
    unknown["request_digest"] = request_digest(unknown)
    with TestClient(app) as client:
        invalid = post(client, bad_decimal)
        mismatch = post(client, bad_digest)
        absent = post(client, unknown)
    assert invalid.status_code == 422 and invalid.json()["code"] == "INVALID_PREDICTION_REQUEST"
    assert mismatch.status_code == 409 and mismatch.json()["code"] == "REQUEST_DIGEST_MISMATCH"
    assert absent.status_code == 404 and absent.json() == {
        "code": "MODEL_NOT_FOUND",
        "message": "Requested model is unavailable.",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "00000000-0000-4000-8000-000000000002"),
        ("model_profile_id", "wrong-profile"),
        ("meas_code", "wrong-meas"),
    ],
)
def test_v2_unknown_exact_identity_boundaries_are_model_not_found(
    monkeypatch, tmp_path: Path, field: str, value: str
) -> None:
    """Relaxing any catalog identity component would select a wrong model fixture."""
    configure_runtime(monkeypatch, tmp_path)
    from valeo_pdm.api.app import app

    body = payload()
    body[field] = value
    body["request_digest"] = request_digest(body)
    with TestClient(app) as client:
        response = post(client, body)
    assert response.status_code == 404
    assert response.json() == {
        "code": "MODEL_NOT_FOUND",
        "message": "Requested model is unavailable.",
    }


def test_v2_unconfigured_catalog_is_safe_503_after_authentication(monkeypatch) -> None:
    """Treating absent runtime catalog as a model miss would hide an unsafe provider state."""
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN", TOKEN)
    monkeypatch.delenv("VALEO_PDM_PREDICTION_V2_MANIFEST", raising=False)
    from valeo_pdm.api.app import app

    with TestClient(app) as client:
        response = post(client, payload())
        health = client.get("/healthz")
    assert response.status_code == 503
    assert response.json() == {
        "code": "PREDICTION_CATALOG_NOT_READY",
        "message": "Prediction catalog is not ready.",
    }
    assert health.status_code == 200


def test_v2_service_failure_never_leaks_path_or_secret(monkeypatch, tmp_path: Path) -> None:
    """Leaking dependency exceptions would expose internal prediction runtime details."""
    configure_runtime(monkeypatch, tmp_path)
    from valeo_pdm.api.app import app
    from valeo_pdm.api.prediction_v2 import get_prediction_service

    class ExplodingService:
        def predict(self, *_args, **_kwargs):
            raise RuntimeError("secret token at /private/v2/model.json")

    app.dependency_overrides[get_prediction_service] = lambda: ExplodingService()
    try:
        with TestClient(app) as client:
            response = post(client, payload())
    finally:
        app.dependency_overrides.pop(get_prediction_service, None)
    assert response.status_code == 500
    assert response.json() == {
        "code": "PREDICTION_FAILED",
        "message": "Prediction could not be completed.",
    }
    assert "/private/v2/model.json" not in response.text and "secret token" not in response.text


def test_legacy_predict_failure_does_not_leak_absolute_path_or_exception(monkeypatch) -> None:
    """Returning legacy exception text would expose checkpoint paths and sensitive upstream details."""
    from valeo_pdm.api import router
    from valeo_pdm.api.app import app

    monkeypatch.setattr(
        router,
        "get_model_config",
        lambda *_: {
            "model_type": "informer",
            "source": "csv",
            "data_path": "fixture.csv",
            "freq": "1min",
            "days_back": 1,
        },
    )
    monkeypatch.setattr(router, "_resolve_csv_path", lambda *_: "fixture.csv")
    monkeypatch.setattr(
        router, "resolve_checkpoint_path", lambda *_: Path("/private/checkpoint.pt")
    )
    monkeypatch.setattr(Path, "exists", lambda self: True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("secret database password at /private/checkpoint.pt")

    monkeypatch.setitem(router.MODEL_PREDICT_FUNCS, "informer", explode)
    with TestClient(app) as client:
        response = client.post(
            "/measPredict/predict", json={"EquipmentCode": "EQ", "MeasCode": "M"}
        )
    assert response.status_code == 500
    encoded = response.text
    assert "/private/checkpoint.pt" not in encoded
    assert "secret database password" not in encoded
