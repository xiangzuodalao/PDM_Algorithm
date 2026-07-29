from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

import pytest
import yaml


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


def write_catalog(
    tmp_path: Path, *, artifact_path: str = "fixture.json", digest: str | None = None
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "fixture.json").write_bytes(ARTIFACT)
    manifest = {
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
                "artifact_path": artifact_path,
                "artifact_sha256": digest or hashlib.sha256(ARTIFACT).hexdigest(),
            }
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest), encoding="utf-8")
    return path


def catalog(tmp_path: Path, **kwargs):
    module = require_module("valeo_pdm.prediction_v2.catalog", "v2 catalog-resolution")
    path = write_catalog(tmp_path, **kwargs)
    return module.ModelCatalog.from_manifest(
        path, object_root=tmp_path, allowed_tenant_ids={TENANT}
    )


def test_catalog_requires_exact_tenant_profile_model_and_measurement(tmp_path: Path) -> None:
    """Relaxing identity comparison would resolve an artifact for the wrong tenant or model."""
    result = catalog(tmp_path).resolve(
        TENANT, "pilot-cnc-vibration", "pilot-fixture-v1-cnc-vibration", "vibration_rms"
    )
    assert (
        result.artifact_sha256 == "5feeb31058fe0521f94758faa22214afe4619e466e22cbb8c6fb7ffcf2369562"
    )
    with pytest.raises(LookupError):
        catalog(tmp_path).resolve(TENANT, "pilot-cnc-vibration", "wrong", "vibration_rms")
    with pytest.raises(LookupError):
        catalog(tmp_path).resolve(
            TENANT, "wrong-profile", "pilot-fixture-v1-cnc-vibration", "vibration_rms"
        )
    with pytest.raises(LookupError):
        catalog(tmp_path).resolve(
            TENANT, "pilot-cnc-vibration", "pilot-fixture-v1-cnc-vibration", "wrong-meas"
        )
    with pytest.raises(LookupError):
        catalog(tmp_path).resolve(
            "00000000-0000-4000-8000-000000000002",
            "pilot-cnc-vibration",
            "pilot-fixture-v1-cnc-vibration",
            "vibration_rms",
        )


@pytest.mark.parametrize("path", ["/tmp/fixture.json", "../fixture.json"])
def test_catalog_fails_closed_for_absolute_and_parent_paths(tmp_path: Path, path: str) -> None:
    """Accepting untrusted manifest paths would permit artifact-root escape."""
    with pytest.raises(ValueError):
        catalog(tmp_path, artifact_path=path)


def test_catalog_fails_closed_for_duplicate_missing_hash_and_escaping_symlink(
    tmp_path: Path,
) -> None:
    """Weak artifact validation would admit ambiguous or unverified prediction fixtures."""
    module = require_module("valeo_pdm.prediction_v2.catalog", "v2 catalog-resolution")
    manifest = write_catalog(tmp_path)
    data = yaml.safe_load(manifest.read_text())
    data["entries"].append(data["entries"][0].copy())
    manifest.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError):
        module.ModelCatalog.from_manifest(
            manifest, object_root=tmp_path, allowed_tenant_ids={TENANT}
        )
    missing = write_catalog(tmp_path / "missing", artifact_path="none.json")
    with pytest.raises(ValueError):
        module.ModelCatalog.from_manifest(
            missing, object_root=missing.parent, allowed_tenant_ids={TENANT}
        )
    wrong = write_catalog(tmp_path / "wrong", digest="0" * 64)
    with pytest.raises(ValueError):
        module.ModelCatalog.from_manifest(
            wrong, object_root=wrong.parent, allowed_tenant_ids={TENANT}
        )
    escape = tmp_path / "escape"
    escape.mkdir()
    manifest = write_catalog(escape)
    (escape / "fixture.json").unlink()
    (escape / "fixture.json").symlink_to(Path("/etc/passwd"))
    with pytest.raises(ValueError):
        module.ModelCatalog.from_manifest(manifest, object_root=escape, allowed_tenant_ids={TENANT})


def test_catalog_rejects_pt_before_reading_artifact_bytes(tmp_path: Path, monkeypatch) -> None:
    """Allowing a hash-valid checkpoint would reintroduce model artifact loading into v2."""
    module = require_module("valeo_pdm.prediction_v2.catalog", "v2 catalog-resolution")
    (tmp_path / "fixture.pt").write_bytes(ARTIFACT)
    manifest = write_catalog(
        tmp_path, artifact_path="fixture.pt", digest=hashlib.sha256(ARTIFACT).hexdigest()
    )
    original = Path.read_bytes

    def reject_pt(path: Path) -> bytes:
        if path.suffix == ".pt":
            raise AssertionError("checkpoint bytes must never be read")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", reject_pt)
    with pytest.raises(ValueError):
        module.ModelCatalog.from_manifest(
            manifest, object_root=tmp_path, allowed_tenant_ids={TENANT}
        )


def test_catalog_fixture_manifest_loads_and_allowlist_is_canonical(tmp_path: Path) -> None:
    """A standalone fixture or permissive tenant parser would weaken independent delivery."""
    module = require_module("valeo_pdm.prediction_v2.catalog", "v2 catalog-resolution")
    fixtures = Path(__file__).parent / "fixtures" / "prediction_v2"
    (tmp_path / "fixture.json").write_bytes((fixtures / "fixture.json").read_bytes())
    catalog = module.ModelCatalog.from_manifest(
        fixtures / "manifest.yaml", object_root=tmp_path, allowed_tenant_ids={TENANT}
    )
    assert catalog.resolve(
        TENANT, "pilot-cnc-vibration", "pilot-fixture-v1-cnc-vibration", "vibration_rms"
    )
    for invalid in (
        "",
        "00000000-0000-4000-8000-00000000000A",
        TENANT.replace("-", ""),
        "{" + TENANT + "}",
    ):
        with pytest.raises(ValueError):
            module.ModelCatalog.from_manifest(
                fixtures / "manifest.yaml", object_root=tmp_path, allowed_tenant_ids={invalid}
            )
