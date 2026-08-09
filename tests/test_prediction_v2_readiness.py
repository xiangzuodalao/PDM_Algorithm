from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from valeo_pdm.prediction_v2.fixture_generator import generate_isolated_fixtures


COMPONENT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = COMPONENT_ROOT / "configs" / "isolated_fixture_manifest.yaml"
TENANT = "00000000-0000-4000-8000-000000000001"
TOKEN = "isolated-pilot-token"


def configure_isolated_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    output_root = tmp_path / "runtime"
    generate_isolated_fixtures(MANIFEST, output_root)
    monkeypatch.setenv("VALEO_PDM_ISOLATED_FIXTURE_MODE", "1")
    monkeypatch.setenv("VALEO_PDM_ALLOWED_TENANT_IDS", TENANT)
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN", TOKEN)
    monkeypatch.setenv(
        "VALEO_PDM_PREDICTION_V2_MANIFEST", str(output_root / "manifest.runtime.yaml")
    )
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_OBJECT_ROOT", str(output_root / "objects"))
    return output_root


def readiness_status(app, *, lifespan: bool = True) -> int:
    if lifespan:
        with TestClient(app) as client:
            return client.get("/readyz").status_code
    return app.routes[-1].endpoint().status_code


def test_readyz_is_healthy_only_for_all_six_unchanged_fixed_fixtures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing fixture must take the isolated runtime out of service."""
    output_root = configure_isolated_runtime(monkeypatch, tmp_path)
    from valeo_pdm.api.app import app

    assert readiness_status(app) == 200
    (output_root / "objects" / "pilot-cnc-vibration.json").unlink()
    assert readiness_status(app) == 503


def test_readyz_rejects_a_byte_tampered_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A changed fixture byte must not pass readiness using the manifest's old digest."""
    output_root = configure_isolated_runtime(monkeypatch, tmp_path)
    artifact = output_root / "objects" / "pilot-cnc-vibration.json"
    artifact.write_bytes(artifact.read_bytes() + b"x")
    from valeo_pdm.api.app import app

    assert readiness_status(app) == 503


def test_readyz_requires_the_generated_runtime_manifest_and_sibling_objects_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Independently supplied source manifests or object roots must not become a runtime catalog."""
    output_root = configure_isolated_runtime(monkeypatch, tmp_path)
    alternate_objects = tmp_path / "alternate-objects"
    shutil.copytree(output_root / "objects", alternate_objects)
    from valeo_pdm.api.app import app

    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_MANIFEST", str(MANIFEST))
    assert readiness_status(app) == 503
    monkeypatch.setenv(
        "VALEO_PDM_PREDICTION_V2_MANIFEST", str(output_root / "manifest.runtime.yaml")
    )
    monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_OBJECT_ROOT", str(alternate_objects))
    assert readiness_status(app) == 503


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("unit", "cm/s"),
        ("value_scale", 9),
        ("request_window_points", 65),
        ("context_points", 59),
        ("horizon_points", 14),
        ("model_profile_id", "pilot-cnc-vibration-altered"),
        ("model_info_id", "pilot-fixture-v1-cnc-vibration-altered"),
        ("preprocessing_version", "pdm-v2-pilot-2"),
    ],
)
def test_readyz_rejects_each_mutated_profile_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, field: str, replacement: object
) -> None:
    """A manifest edited after generation must not be accepted by readiness."""
    output_root = configure_isolated_runtime(monkeypatch, tmp_path)
    runtime_path = output_root / "manifest.runtime.yaml"
    runtime = yaml.safe_load(runtime_path.read_text(encoding="utf-8"))
    runtime["entries"][0][field] = replacement
    runtime_path.write_text(yaml.safe_dump(runtime), encoding="utf-8")
    from valeo_pdm.api.app import app

    assert readiness_status(app) == 503


@pytest.mark.parametrize(
    "allowlist", ["", "*", "not-a-uuid", "00000000-0000-4000-8000-00000000000A"]
)
def test_isolated_mode_rejects_empty_wildcard_or_invalid_tenant_allowlists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, allowlist: str
) -> None:
    """An unvalidated tenant allowlist would permit cross-tenant fixture use."""
    configure_isolated_runtime(monkeypatch, tmp_path)
    monkeypatch.setenv("VALEO_PDM_ALLOWED_TENANT_IDS", allowlist)
    from valeo_pdm.api.app import app

    with pytest.raises(RuntimeError, match="tenant"):
        readiness_status(app)


@pytest.mark.parametrize("token", [None, "   "])
def test_isolated_mode_requires_a_nonblank_bearer_token_and_can_never_be_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, token: str | None
) -> None:
    """A tokenless isolated catalog must fail closed before it can report ready."""
    configure_isolated_runtime(monkeypatch, tmp_path)
    if token is None:
        monkeypatch.delenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN")
    else:
        monkeypatch.setenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN", token)
    from valeo_pdm.api.app import app

    assert readiness_status(app, lifespan=False) == 503
    with pytest.raises(RuntimeError, match="bearer"):
        readiness_status(app)


def test_isolated_pilot_manifest_requires_isolated_fixture_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pilot fixture manifest outside isolated mode would weaken deployment safeguards."""
    output_root = configure_isolated_runtime(monkeypatch, tmp_path)
    monkeypatch.delenv("VALEO_PDM_ISOLATED_FIXTURE_MODE")
    monkeypatch.setenv(
        "VALEO_PDM_PREDICTION_V2_MANIFEST", str(output_root / "manifest.runtime.yaml")
    )
    from valeo_pdm.api.app import app

    with pytest.raises(RuntimeError, match="isolated"):
        readiness_status(app)
