from __future__ import annotations

import hashlib
from pathlib import Path

import rfc8785
import yaml

from valeo_pdm.prediction_v2.models import ModelProfile


def build_fixture_artifact(profile: ModelProfile) -> bytes:
    return rfc8785.dumps(
        {
            "kind": "repeat-last",
            "model_info_id": profile.model_info_id,
            "model_profile_id": profile.model_profile_id,
            "preprocessing_version": profile.preprocessing_version,
            "schema_version": 1,
        }
    )


def generate_isolated_fixtures(manifest_path: Path, output_root: Path) -> dict[str, str]:
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("fixture_mode") != "isolated-pilot":
        raise ValueError("isolated pilot fixture manifest is required")
    entries = raw.get("entries")
    if not isinstance(entries, list) or len(entries) != 6:
        raise ValueError("exactly six isolated fixture entries are required")
    _prepare_output_root(output_root)
    objects = output_root / "objects"
    objects.mkdir(mode=0o700)
    runtime_entries: list[dict[str, object]] = []
    generated: dict[str, str] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("invalid isolated fixture entry")
        profile = ModelProfile.model_validate(
            {
                key: item[key]
                for key in (
                    "model_profile_id",
                    "model_info_id",
                    "meas_code",
                    "unit",
                    "value_scale",
                    "sampling_frequency",
                    "request_window_points",
                    "context_points",
                    "horizon_points",
                    "preprocessing_version",
                )
            }
        )
        artifact = build_fixture_artifact(profile)
        artifact_sha256 = hashlib.sha256(artifact).hexdigest()
        if artifact_sha256 != item.get("artifact_sha256"):
            raise ValueError("isolated fixture artifact does not match its fixed hash")
        artifact_path = f"{profile.model_profile_id}.json"
        with (objects / artifact_path).open("xb") as stream:
            stream.write(artifact)
        runtime_entry = dict(item)
        runtime_entry["artifact_path"] = artifact_path
        runtime_entries.append(runtime_entry)
        generated[profile.model_profile_id] = artifact_sha256
    runtime = {"fixture_mode": "isolated-pilot", "entries": runtime_entries}
    with (output_root / "manifest.runtime.yaml").open("x", encoding="utf-8") as stream:
        yaml.safe_dump(runtime, stream, allow_unicode=True, sort_keys=False)
    return generated


def _prepare_output_root(output_root: Path) -> None:
    if output_root.exists() or output_root.is_symlink():
        if output_root.is_symlink():
            raise ValueError("isolated fixture output root must not be a symlink")
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise ValueError("isolated fixture output root must be empty")
        return
    output_root.mkdir(mode=0o700, parents=True)
