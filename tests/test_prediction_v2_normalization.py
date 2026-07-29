from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError


FIXTURES = Path(__file__).parent / "fixtures" / "prediction_v2"
REQUEST_DIGEST = "067096fc115546164d8608cddadc56b185fefea9dffab4c7462c8d28011cdfbd"
INPUT_DIGEST = "c67fe58c14f2e8a1ef252e6ff47b4877c0da8fa52664772dd3c08ae3d6552609"


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


def request_payload() -> dict:
    return json.loads((FIXTURES / "request.json").read_text(encoding="utf-8"))


def profile():
    models = require_module("valeo_pdm.prediction_v2.models", "v2 strict models")
    return models.ModelProfile(
        model_profile_id="pilot-cnc-vibration",
        model_info_id="pilot-fixture-v1-cnc-vibration",
        meas_code="vibration_rms",
        unit="mm/s",
        value_scale=2,
        sampling_frequency="1min",
        request_window_points=66,
        context_points=60,
        horizon_points=15,
        preprocessing_version="pdm-v2-pilot-1",
    )


def request(payload: dict):
    models = require_module("valeo_pdm.prediction_v2.models", "v2 strict models")
    return models.PredictionRequestV2.model_validate(payload)


def test_canonical_decimal_is_fixed_scale_and_half_even() -> None:
    """Removing Decimal quantization would change replay-stable normalized values."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 canonical-decimal normalization"
    )
    assert normalization.canonical_decimal("-0.000", scale=2) == "0.00"
    assert normalization.canonical_decimal("1.245", scale=2) == "1.24"
    assert normalization.canonical_decimal("1.255", scale=2) == "1.26"
    with pytest.raises(ValueError):
        normalization.canonical_decimal("9" * 1_000, scale=2)


@pytest.mark.parametrize("value", ["1e2", "NaN", "Infinity", "-Infinity", "+1.00"])
def test_canonical_decimal_rejects_non_contract_lexemes(value: str) -> None:
    """Accepting exponent or non-finite input would make canonical request identity ambiguous."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 canonical-decimal normalization"
    )
    with pytest.raises(ValueError):
        normalization.canonical_decimal(value, scale=2)


def test_request_projection_sorts_history_and_excludes_replay_metadata() -> None:
    """Including correlation or supplied digest would make equivalent retry input hash differently."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 request digest normalization"
    )
    payload = request_payload()
    payload["history"] = list(reversed(payload["history"]))
    original = request(payload)
    changed = request(
        {
            **payload,
            "correlation_id": "00000000-0000-4000-8000-000000000103",
            "request_digest": "f" * 64,
        }
    )
    projection = normalization.request_projection(original)
    assert "correlation_id" not in projection and "request_digest" not in projection
    assert [point["data_id"] for point in projection["history"][:2]] == [
        "fixture-000",
        "fixture-001",
    ]
    assert normalization.calculate_request_digest(
        original
    ) == normalization.calculate_request_digest(changed)
    assert normalization.calculate_request_digest(original) == REQUEST_DIGEST


def test_request_projection_orders_same_timestamp_by_data_id() -> None:
    """Changing tie ordering would produce a different cross-language replay digest."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 request digest normalization"
    )
    payload = request_payload()
    first = payload["history"][0]
    payload["history"] = [
        {**first, "data_id": "z"},
        {**first, "data_id": "a"},
        *payload["history"][1:],
    ]
    assert [
        point["data_id"]
        for point in normalization.request_projection(request(payload))["history"][:2]
    ] == [
        "a",
        "z",
    ]


def test_normalization_nfc_epoch_buckets_and_exact_golden_input_digest() -> None:
    """Changing canonical unit, bucketing, or digest projection would break the frozen provider example."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 input normalization"
    )
    payload = request_payload()
    payload["unit"] = "mm/s"
    normalized = normalization.normalize_history(request(payload), profile(), now_ms=1785287700000)
    assert normalized.values == ("4.00",) * 66
    assert normalized.last_bucket_ms == 1785287640000
    assert normalized.input_digest == INPUT_DIGEST
    payload["unit"] = " A\u030a "
    payload["history"] = [{**point, "unit": " A\u030a "} for point in payload["history"]]
    nfc_profile = profile().model_copy(update={"unit": "Å"})
    assert (
        normalization.normalize_history(request(payload), nfc_profile, now_ms=1785287700000).unit
        == "Å"
    )


def test_request_digest_uses_profile_normalized_units_and_decimals_before_verification() -> None:
    """Hashing raw spacing or scale would reject a canonical replay identity."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 canonical request normalization"
    )
    payload = request_payload()
    payload["unit"] = " mm/s "
    payload["history"] = [
        {**point, "unit": " mm/s ", "value": "4.000"} for point in payload["history"]
    ]
    canonical = normalization.canonicalize_request(request(payload), profile())
    assert canonical.unit == "mm/s"
    assert canonical.history[0].value == "4.00"
    assert normalization.calculate_request_digest(canonical) == REQUEST_DIGEST


def test_normalization_aggregates_duplicates_and_inserts_explicit_null_buckets() -> None:
    """Skipping duplicate half-even averaging or missing buckets changes the model input."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 input normalization"
    )
    payload = request_payload()
    payload["history"] = [
        {**payload["history"][0], "data_id": "b", "value": "1.24"},
        {**payload["history"][0], "data_id": "a", "value": "1.25"},
        *payload["history"][2:],
    ]
    normalized = normalization.normalize_history(request(payload), profile(), now_ms=1785287700000)
    assert normalized.values[0] == "1.24"
    assert normalized.values[1] is None


def test_normalization_accepts_six_missing_and_rejects_duplicate_identity() -> None:
    """Changing missing or identity limits would admit an invalid model window."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 input normalization"
    )
    payload = request_payload()
    payload["history"] = [
        point for index, point in enumerate(payload["history"]) if index not in {0, 2, 4, 6, 64, 65}
    ]
    normalized = normalization.normalize_history(request(payload), profile(), now_ms=1785287700000)
    assert normalized.values[0] is None and normalized.values[-2:] == (None, None)
    duplicate = request_payload()
    duplicate["history"].append(duplicate["history"][0].copy())
    with pytest.raises(ValueError):
        normalization.normalize_history(request(duplicate), profile(), now_ms=1785287700000)


@pytest.mark.parametrize("mutation", ["too_many", "consecutive", "future"])
def test_normalization_rejects_invalid_missingness_and_future_history(mutation: str) -> None:
    """Relaxing sparse or future data guards would allow invalid model windows."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 input normalization"
    )
    payload = request_payload()
    if mutation == "too_many":
        payload["history"] = payload["history"][7:]
    elif mutation == "consecutive":
        payload["history"] = [
            p for index, p in enumerate(payload["history"]) if index not in {3, 4, 5}
        ]
    else:
        payload["history"][0]["timestamp"] = 1785287700000
    with pytest.raises(ValueError):
        normalization.normalize_history(request(payload), profile(), now_ms=1785287700000)


@pytest.mark.parametrize("mutation", ["equal_window", "unaligned", "wrong_count", "outside"])
def test_normalization_rejects_window_and_history_boundaries(mutation: str) -> None:
    """Relaxing aligned 66-point window bounds would change bucket identity."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 input normalization"
    )
    payload = request_payload()
    if mutation == "equal_window":
        payload["window_end"] = payload["window_start"]
    elif mutation == "unaligned":
        payload["window_start"] += 1
    elif mutation == "wrong_count":
        payload["window_end"] -= 60_000
    else:
        payload["history"][0]["timestamp"] = payload["window_start"] - 60_000
    with pytest.raises(ValueError):
        normalization.normalize_history(request(payload), profile(), now_ms=1785287700000)


def test_normalization_rejects_actual_future_clock_boundary() -> None:
    """Accepting a window ending after the injected clock would allow future observations."""
    normalization = require_module(
        "valeo_pdm.prediction_v2.normalization", "v2 input normalization"
    )
    with pytest.raises(ValueError):
        normalization.normalize_history(request(request_payload()), profile(), now_ms=1785287699999)


def test_strict_models_accept_only_canonical_uuid_and_integer_json_tokens() -> None:
    """Permissive UUID or numeric coercion would weaken the HTTP contract boundary."""
    payload = request_payload()
    assert request(payload).tenant_id.hex == "00000000000040008000000000000001"
    for field, invalid in (
        ("tenant_id", "00000000-0000-4000-8000-00000000000A"),
        ("window_start", 1785283740000.0),
        ("window_end", True),
    ):
        changed = {**payload, field: invalid}
        with pytest.raises(ValidationError):
            request(changed)
