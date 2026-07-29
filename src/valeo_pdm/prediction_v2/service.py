from __future__ import annotations

import hmac
from datetime import UTC, datetime

from valeo_pdm.prediction_v2.catalog import ModelCatalog
from valeo_pdm.prediction_v2.models import PredictionRequestV2, PredictionResponseV2
from valeo_pdm.prediction_v2.normalization import calculate_request_digest, normalize_history
from valeo_pdm.prediction_v2.predictors import RepeatLastFixturePredictor


class RequestDigestMismatch(ValueError):
    """The caller's replay identity does not describe its request."""


class PredictionV2Service:
    def __init__(
        self, catalog: ModelCatalog, predictor: RepeatLastFixturePredictor | None = None
    ) -> None:
        self._catalog = catalog
        self._predictor = predictor or RepeatLastFixturePredictor()

    def predict(self, request: PredictionRequestV2, *, now: datetime) -> PredictionResponseV2:
        expected = calculate_request_digest(request)
        if not hmac.compare_digest(expected, request.request_digest):
            raise RequestDigestMismatch("request digest does not match")
        resolved = self._catalog.resolve(
            request.tenant_id, request.model_profile_id, request.model_info_id, request.meas_code
        )
        if now.tzinfo is None:
            raise ValueError("now must be timezone-aware")
        now_ms = int(now.astimezone(UTC).timestamp() * 1000)
        normalized = normalize_history(request, resolved.profile, now_ms=now_ms)
        forecast = self._predictor.predict(normalized, resolved.profile)
        return PredictionResponseV2(
            correlation_id=request.correlation_id,
            equipment_id=request.equipment_id,
            model_profile_id=resolved.profile.model_profile_id,
            model_info_id=resolved.profile.model_info_id,
            model_artifact_sha256=resolved.artifact_sha256,
            meas_code=resolved.profile.meas_code,
            request_digest=expected,
            input_digest=normalized.input_digest,
            generated_at=now.astimezone(UTC),
            forecast=forecast,
        )
