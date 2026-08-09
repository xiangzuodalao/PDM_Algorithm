from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from typing import Any

import rfc8785

from valeo_pdm.prediction_v2.models import ModelProfile, PredictionRequestV2


DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


@dataclass(frozen=True)
class NormalizedInput:
    values: tuple[str | None, ...]
    last_bucket_ms: int
    unit: str
    input_digest: str


def canonical_decimal(value: str, *, scale: int) -> str:
    if type(value) is not str or DECIMAL_RE.fullmatch(value) is None:
        raise ValueError("decimal string must be finite and non-exponent")
    try:
        decimal = Decimal(value)
        quantized = decimal.quantize(Decimal(1).scaleb(-scale), rounding=ROUND_HALF_EVEN)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal") from exc
    if quantized == 0:
        quantized = abs(quantized)
    return f"{quantized:.{scale}f}"


def canonical_unit(value: str) -> str:
    if type(value) is not str:
        raise ValueError("unit must be a string")
    unit = unicodedata.normalize("NFC", value.strip())
    if not unit:
        raise ValueError("unit must be non-empty")
    return unit


def _canonical_bytes(value: Any) -> bytes:
    return rfc8785.dumps(value)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def request_projection(request: PredictionRequestV2) -> dict[str, Any]:
    history = sorted(
        (
            {
                "data_id": point.data_id,
                "timestamp": point.timestamp,
                "value": point.value,
                "unit": point.unit,
            }
            for point in request.history
        ),
        key=lambda point: (point["timestamp"], point["data_id"]),
    )
    return {
        "tenant_id": str(request.tenant_id),
        "equipment_id": str(request.equipment_id),
        "model_profile_id": request.model_profile_id,
        "model_info_id": request.model_info_id,
        "meas_code": request.meas_code,
        "unit": request.unit,
        "sampling_frequency": request.sampling_frequency,
        "window_start": request.window_start,
        "window_end": request.window_end,
        "history": history,
    }


def calculate_request_digest(request: PredictionRequestV2) -> str:
    return _digest(request_projection(request))


def canonicalize_request(
    request: PredictionRequestV2, profile: ModelProfile
) -> PredictionRequestV2:
    """Normalize replay identity fields before its request digest is verified."""
    unit = canonical_unit(request.unit)
    if unit != canonical_unit(profile.unit):
        raise ValueError("unit mismatch")
    history = [
        point.model_copy(
            update={
                "value": canonical_decimal(point.value, scale=profile.value_scale),
                "unit": canonical_unit(point.unit),
            }
        )
        for point in request.history
    ]
    if any(point.unit != unit for point in history):
        raise ValueError("unit mismatch")
    return request.model_copy(update={"unit": unit, "history": history})


def normalize_history(
    request: PredictionRequestV2, profile: ModelProfile, *, now_ms: int
) -> NormalizedInput:
    if type(now_ms) is not int or request.window_start >= request.window_end:
        raise ValueError("invalid window")
    interval = profile.interval_ms
    if request.sampling_frequency != profile.sampling_frequency:
        raise ValueError("sampling frequency mismatch")
    request_unit = canonical_unit(request.unit)
    profile_unit = canonical_unit(profile.unit)
    if request_unit != profile_unit:
        raise ValueError("unit mismatch")
    if (
        request.window_end > now_ms
        or request.window_start % interval
        or request.window_end % interval
    ):
        raise ValueError("window must be aligned and not future")
    if (request.window_end - request.window_start) // interval != profile.request_window_points:
        raise ValueError("window point count mismatch")
    grouped: dict[int, list[Decimal]] = {}
    seen: set[tuple[int, str]] = set()
    for point in request.history:
        if point.timestamp < request.window_start or point.timestamp >= request.window_end:
            raise ValueError("history outside window")
        if point.timestamp > now_ms or point.timestamp % interval:
            raise ValueError("history timestamp is invalid")
        identity = (point.timestamp, point.data_id)
        if identity in seen:
            raise ValueError("duplicate history identity")
        seen.add(identity)
        if canonical_unit(point.unit) != request_unit:
            raise ValueError("unit mismatch")
        normalized_value = canonical_decimal(point.value, scale=profile.value_scale)
        grouped.setdefault(point.timestamp, []).append(Decimal(normalized_value))
    values: list[str | None] = []
    buckets: list[dict[str, str | int | None]] = []
    consecutive_missing = 0
    for index in range(profile.request_window_points):
        timestamp = request.window_start + index * interval
        samples = grouped.get(timestamp)
        if samples:
            mean = sum(samples) / len(samples)
            value = canonical_decimal(format(mean, "f"), scale=profile.value_scale)
            consecutive_missing = 0
        else:
            value = None
            consecutive_missing += 1
        values.append(value)
        buckets.append({"timestamp": timestamp, "value": value})
    missing = values.count(None)
    if missing > profile.request_window_points // 10 or consecutive_missing > 2:
        raise ValueError("history missingness exceeds limit")
    # Catch runs that ended before a non-null bucket as well.
    if any(
        values[index] is None and values[index + 1] is None and values[index + 2] is None
        for index in range(len(values) - 2)
    ):
        raise ValueError("consecutive missing history exceeds limit")
    projection = {
        "model_profile_id": profile.model_profile_id,
        "model_info_id": profile.model_info_id,
        "meas_code": profile.meas_code,
        "preprocessing_version": profile.preprocessing_version,
        "buckets": buckets,
    }
    return NormalizedInput(
        values=tuple(values),
        last_bucket_ms=request.window_end - interval,
        unit=request_unit,
        input_digest=_digest(projection),
    )
