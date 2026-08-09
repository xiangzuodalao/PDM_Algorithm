from __future__ import annotations

from valeo_pdm.prediction_v2.models import ForecastPointV2, ModelProfile
from valeo_pdm.prediction_v2.normalization import NormalizedInput


class RepeatLastFixturePredictor:
    def predict(
        self, normalized: NormalizedInput, profile: ModelProfile
    ) -> tuple[ForecastPointV2, ...]:
        last_value = next(value for value in reversed(normalized.values) if value is not None)
        return tuple(
            ForecastPointV2(
                timestamp=normalized.last_bucket_ms + profile.interval_ms * index,
                value=last_value,
                unit=profile.unit,
            )
            for index in range(1, profile.horizon_points + 1)
        )
