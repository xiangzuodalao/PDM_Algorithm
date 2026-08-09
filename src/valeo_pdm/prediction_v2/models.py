from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


CANONICAL_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CanonicalUuid = Annotated[UUID, Field(strict=False)]


def _canonical_uuid(value: object) -> object:
    if isinstance(value, UUID):
        return value
    if type(value) is not str or CANONICAL_UUID_RE.fullmatch(value) is None:
        raise ValueError("canonical lowercase hyphenated UUID required")
    return value


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class HistoryPointV2(_StrictModel):
    data_id: str = Field(min_length=1)
    timestamp: int = Field(ge=0)
    value: str = Field(min_length=1)
    unit: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def require_int_token(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("integer JSON token required")
        return value


class PredictionRequestV2(_StrictModel):
    tenant_id: CanonicalUuid
    correlation_id: CanonicalUuid
    equipment_id: CanonicalUuid
    model_profile_id: str = Field(min_length=1)
    model_info_id: str = Field(min_length=1)
    meas_code: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    sampling_frequency: str = Field(min_length=1)
    window_start: int = Field(ge=0)
    window_end: int = Field(ge=0)
    request_digest: str = Field(min_length=64, max_length=64)
    history: list[HistoryPointV2]

    @field_validator("tenant_id", "correlation_id", "equipment_id", mode="before")
    @classmethod
    def require_canonical_uuid_string(cls, value: object) -> object:
        return _canonical_uuid(value)

    @field_validator("window_start", "window_end")
    @classmethod
    def require_int_token(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("integer JSON token required")
        return value

    @field_validator("request_digest")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("lowercase SHA-256 required")
        return value


class ModelProfile(_StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    model_profile_id: str
    model_info_id: str
    meas_code: str
    unit: str
    value_scale: int
    sampling_frequency: str
    request_window_points: int
    context_points: int
    horizon_points: int
    preprocessing_version: str

    @property
    def interval_ms(self) -> int:
        frequencies = {"1min": 60_000}
        try:
            return frequencies[self.sampling_frequency]
        except KeyError as exc:
            raise ValueError("unsupported sampling frequency") from exc


class ForecastPointV2(_StrictModel):
    timestamp: int = Field(ge=0)
    value: str = Field(min_length=1)
    unit: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def require_int_token(cls, value: int) -> int:
        if type(value) is not int:
            raise ValueError("integer JSON token required")
        return value


class PredictionResponseV2(_StrictModel):
    correlation_id: CanonicalUuid
    equipment_id: CanonicalUuid
    model_profile_id: str
    model_info_id: str
    model_artifact_sha256: str
    meas_code: str
    request_digest: str
    input_digest: str
    generated_at: datetime
    forecast: tuple[ForecastPointV2, ...]

    @field_validator("correlation_id", "equipment_id", mode="before")
    @classmethod
    def require_canonical_uuid_string(cls, value: object) -> object:
        return _canonical_uuid(value)

    @field_validator("model_artifact_sha256", "request_digest", "input_digest")
    @classmethod
    def require_digest(cls, value: str) -> str:
        if SHA256_RE.fullmatch(value) is None:
            raise ValueError("lowercase SHA-256 required")
        return value

    @field_validator("forecast")
    @classmethod
    def require_horizon(cls, value: tuple[ForecastPointV2, ...]) -> tuple[ForecastPointV2, ...]:
        if len(value) != 15:
            raise ValueError("forecast must contain exactly 15 points")
        return value
