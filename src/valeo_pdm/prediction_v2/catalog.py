from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import UUID

import yaml
import rfc8785

from valeo_pdm.prediction_v2.models import CANONICAL_UUID_RE, ModelProfile


_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd
_FIXED_ARTIFACT_HASHES = {
    "pilot-cnc-vibration": "5feeb31058fe0521f94758faa22214afe4619e466e22cbb8c6fb7ffcf2369562",
    "pilot-injection-pressure": "6e3faef72b69bbd7e9562e871f405a38286075461d1b0c3a5d170c72d7e0f1a9",
    "pilot-robot-position": "11f162e58f9958ed405a8c67dcbb780900321385d8c2eb45122367b11ed5016e",
    "pilot-tightening-torque": "dae2043bc142ad6bb984e296d2cff21f6a93ccba49f1e42c02ca26b6e413d014",
    "pilot-compressor-pressure": "5e2959d7617adb4fad4da45702bbef75f11832f8f7f9818c7f2e662e83bccf38",
    "pilot-eol-pass-rate": "5fe0b2a40d796cbbd5e42d67cce20d195c2c67a75795cc9f4390848f1c119377",
}
_FIXED_PROFILE_CONFIG_HASHES = {
    "pilot-cnc-vibration": "4db612fb9af5095479fc5ecd03e2118d848f79e5c149073fce2df10cf34c64c1",
    "pilot-injection-pressure": "9c37acf828421f27f25b32c10453ff74c0108993822af2997e5379879406b390",
    "pilot-robot-position": "a33b9a84f88c2c5c78d99a0ccd222422c145cb495e4fe8c5b289010f4e768368",
    "pilot-tightening-torque": "43c9a68832b2030051dca0a320c13bc526b8ae58d810bc0e44052872d836cac8",
    "pilot-compressor-pressure": "6c955b29105ce34724dac26280ee2ff3f428e55c5e5269f40c6e32860ccfa1e8",
    "pilot-eol-pass-rate": "61f0d0323b511999351203cc0017e5c9cd5e646fa7366a3616607c01b781650b",
}


@dataclass(frozen=True)
class ResolvedModel:
    profile: ModelProfile
    artifact_sha256: str


class ModelCatalog:
    def __init__(
        self,
        entries: dict[tuple[str, str, str], ResolvedModel],
        allowed_tenant_ids: set[str],
        *,
        manifest_path: Path,
        object_root: Path,
    ):
        self._entries = entries
        self._allowed_tenant_ids = allowed_tenant_ids
        self._manifest_path = manifest_path
        self._object_root = object_root

    @classmethod
    def from_manifest(
        cls,
        manifest_path: Path,
        *,
        object_root: Path,
        allowed_tenant_ids: Iterable[str],
    ) -> ModelCatalog:
        try:
            root = object_root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("prediction object root is unavailable") from exc
        if not root.is_dir():
            raise ValueError("prediction object root must be a directory")
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        fixture_mode = raw.get("fixture_mode") if isinstance(raw, dict) else None
        isolated_pilot = fixture_mode == "isolated-pilot"
        if fixture_mode is not None and not isolated_pilot:
            raise ValueError("unsupported prediction fixture mode")
        if isolated_pilot and not isolated_fixture_mode_enabled():
            raise ValueError("isolated fixture mode is not enabled")
        entries = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("prediction catalog entries are required")
        if isolated_pilot and len(entries) != len(_FIXED_ARTIFACT_HASHES):
            raise ValueError("isolated fixture catalog requires six entries")
        root_fd = _open_object_root(root)
        resolved: dict[tuple[str, str, str], ResolvedModel] = {}
        try:
            for item in entries:
                if not isinstance(item, dict):
                    raise ValueError("invalid prediction catalog entry")
                try:
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
                    artifact_path = item["artifact_path"]
                    expected_hash = item["artifact_sha256"]
                    kind = item.get("kind")
                    profile_config_sha256 = item.get("profile_config_sha256")
                except (KeyError, ValueError) as exc:
                    raise ValueError("invalid prediction catalog entry") from exc
                if (
                    profile.request_window_points != 66
                    or profile.context_points != 60
                    or profile.horizon_points != 15
                ):
                    raise ValueError("unsupported prediction profile shape")
                if (
                    type(artifact_path) is not str
                    or not artifact_path
                    or Path(artifact_path).is_absolute()
                ):
                    raise ValueError("artifact path must be relative")
                relative = Path(artifact_path)
                if (
                    relative.name != artifact_path
                    or ".." in relative.parts
                    or relative.suffix != ".json"
                ):
                    raise ValueError("artifact must be a single relative JSON fixture")
                actual_hash = _digest_regular_artifact(root_fd, artifact_path)
                if actual_hash != expected_hash:
                    raise ValueError("prediction artifact hash mismatch")
                if isolated_pilot:
                    _validate_fixed_isolated_entry(
                        profile, kind, expected_hash, profile_config_sha256
                    )
                identity = (profile.model_profile_id, profile.model_info_id, profile.meas_code)
                if identity in resolved:
                    raise ValueError("duplicate prediction model identity")
                resolved[identity] = ResolvedModel(profile=profile, artifact_sha256=actual_hash)
        finally:
            os.close(root_fd)
        allowed: set[str] = set()
        for tenant in allowed_tenant_ids:
            if type(tenant) is not str or CANONICAL_UUID_RE.fullmatch(tenant) is None:
                raise ValueError("tenant allowlist requires canonical UUIDs")
            allowed.add(str(UUID(tenant)))
        if isolated_pilot and set(_FIXED_ARTIFACT_HASHES) != {
            item.profile.model_profile_id for item in resolved.values()
        }:
            raise ValueError("isolated fixture catalog identities do not match")
        return cls(
            resolved,
            allowed,
            manifest_path=manifest_path,
            object_root=root,
        )

    def readiness(self) -> bool:
        try:
            ModelCatalog.from_manifest(
                self._manifest_path,
                object_root=self._object_root,
                allowed_tenant_ids=self._allowed_tenant_ids,
            )
        except (OSError, ValueError, yaml.YAMLError):
            return False
        return True

    def resolve(
        self,
        tenant_id: UUID | str,
        model_profile_id: str,
        model_info_id: str,
        meas_code: str,
    ) -> ResolvedModel:
        if str(tenant_id) not in self._allowed_tenant_ids:
            raise LookupError("model identity is unavailable")
        try:
            return self._entries[(model_profile_id, model_info_id, meas_code)]
        except KeyError as exc:
            raise LookupError("model identity is unavailable") from exc


def isolated_fixture_mode_enabled() -> bool:
    return os.getenv("VALEO_PDM_ISOLATED_FIXTURE_MODE") == "1"


def validate_isolated_runtime_environment() -> None:
    if not isolated_fixture_mode_enabled():
        return
    if not os.getenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN", "").strip():
        raise ValueError("isolated fixture mode requires a bearer token")
    tenant_ids = os.getenv("VALEO_PDM_ALLOWED_TENANT_IDS", "").split(",")
    if not tenant_ids or not all(CANONICAL_UUID_RE.fullmatch(tenant) for tenant in tenant_ids):
        raise ValueError("isolated fixture mode requires canonical tenant UUIDs")


def manifest_is_isolated_pilot(manifest_path: Path) -> bool:
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError("isolated fixture manifest is unavailable") from exc
    return isinstance(raw, dict) and raw.get("fixture_mode") == "isolated-pilot"


def _validate_fixed_isolated_entry(
    profile: ModelProfile, kind: object, artifact_sha256: object, profile_config_sha256: object
) -> None:
    profile_id = profile.model_profile_id
    if kind != "repeat-last":
        raise ValueError("unsupported isolated fixture kind")
    if artifact_sha256 != _FIXED_ARTIFACT_HASHES.get(profile_id):
        raise ValueError("isolated fixture artifact hash mismatch")
    expected_config_hash = _FIXED_PROFILE_CONFIG_HASHES.get(profile_id)
    if profile_config_sha256 != expected_config_hash:
        raise ValueError("isolated fixture profile hash mismatch")
    projection = {
        "context_points": profile.context_points,
        "horizon_points": profile.horizon_points,
        "kind": kind,
        "meas_code": profile.meas_code,
        "model_artifact_sha256": artifact_sha256,
        "model_info_id": profile.model_info_id,
        "model_profile_id": profile.model_profile_id,
        "preprocessing_version": profile.preprocessing_version,
        "request_window_points": profile.request_window_points,
        "sampling_frequency": profile.sampling_frequency,
        "unit": profile.unit,
        "value_scale": profile.value_scale,
    }
    actual_config_hash = hashlib.sha256(rfc8785.dumps(projection)).hexdigest()
    if actual_config_hash != expected_config_hash:
        raise ValueError("isolated fixture profile configuration mismatch")


def _required_open_flags() -> tuple[int, int]:
    required = ("O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC", "O_NONBLOCK")
    if any(not hasattr(os, name) for name in required) or not _OPEN_SUPPORTS_DIR_FD:
        raise ValueError("safe descriptor operations are unavailable")
    return (
        os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK,
    )


def _open_object_root(root: Path) -> int:
    root_flags, _ = _required_open_flags()
    try:
        return os.open(root, os.O_RDONLY | root_flags)
    except OSError as exc:
        raise ValueError("prediction object root is unavailable") from exc


def _digest_regular_artifact(root_fd: int, basename: str) -> str:
    _, artifact_flags = _required_open_flags()
    try:
        artifact_fd = os.open(basename, os.O_RDONLY | artifact_flags, dir_fd=root_fd)
    except OSError as exc:
        raise ValueError("prediction artifact is unavailable") from exc
    try:
        if not stat.S_ISREG(os.fstat(artifact_fd).st_mode):
            raise ValueError("prediction artifact must be a regular file")
        digest = hashlib.sha256()
        while chunk := os.read(artifact_fd, 1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()
    except OSError as exc:
        raise ValueError("prediction artifact cannot be read") from exc
    finally:
        os.close(artifact_fd)
