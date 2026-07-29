from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from uuid import UUID

import yaml

from valeo_pdm.prediction_v2.models import CANONICAL_UUID_RE, ModelProfile


_OPEN_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd


@dataclass(frozen=True)
class ResolvedModel:
    profile: ModelProfile
    artifact_sha256: str


class ModelCatalog:
    def __init__(
        self, entries: dict[tuple[str, str, str], ResolvedModel], allowed_tenant_ids: set[str]
    ):
        self._entries = entries
        self._allowed_tenant_ids = allowed_tenant_ids

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
        entries = raw.get("entries") if isinstance(raw, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError("prediction catalog entries are required")
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
        return cls(resolved, allowed)

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
