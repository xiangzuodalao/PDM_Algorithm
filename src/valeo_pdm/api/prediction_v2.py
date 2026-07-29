from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException

from valeo_pdm.prediction_v2.auth import valid_bearer_token
from valeo_pdm.prediction_v2.catalog import ModelCatalog
from valeo_pdm.prediction_v2.models import PredictionRequestV2, PredictionResponseV2
from valeo_pdm.prediction_v2.service import PredictionV2Service, RequestDigestMismatch


router = APIRouter(prefix="/api/v2", tags=["deterministic-predictions"])


def _error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": code, "message": message})


def authenticate_prediction_v2(authorization: Annotated[str | None, Header()] = None) -> None:
    expected = os.getenv("VALEO_PDM_PREDICTION_V2_BEARER_TOKEN")
    if not valid_bearer_token(authorization, expected):
        raise _error(401, "PREDICTION_UNAUTHORIZED", "Authentication failed.")


def get_prediction_service() -> PredictionV2Service:
    manifest_value = os.getenv("VALEO_PDM_PREDICTION_V2_MANIFEST")
    object_root_value = os.getenv("VALEO_PDM_PREDICTION_V2_OBJECT_ROOT")
    if not manifest_value or not object_root_value:
        raise _error(503, "PREDICTION_CATALOG_NOT_READY", "Prediction catalog is not ready.")
    tenant_ids = os.getenv("VALEO_PDM_ALLOWED_TENANT_IDS", "").split(",")
    try:
        catalog = ModelCatalog.from_manifest(
            Path(manifest_value), object_root=Path(object_root_value), allowed_tenant_ids=tenant_ids
        )
    except Exception as exc:
        raise _error(
            503, "PREDICTION_CATALOG_NOT_READY", "Prediction catalog is not ready."
        ) from exc
    return PredictionV2Service(catalog)


@router.post("/predictions", response_model=PredictionResponseV2)
def create_prediction(
    request: PredictionRequestV2,
    _: Annotated[None, Depends(authenticate_prediction_v2)],
    service: Annotated[PredictionV2Service, Depends(get_prediction_service)],
) -> PredictionResponseV2:
    try:
        return service.predict(request, now=datetime.now(UTC))
    except RequestDigestMismatch as exc:
        raise _error(409, "REQUEST_DIGEST_MISMATCH", "Request digest does not match.") from exc
    except LookupError as exc:
        raise _error(404, "MODEL_NOT_FOUND", "Requested model is unavailable.") from exc
    except ValueError as exc:
        raise _error(422, "INVALID_PREDICTION_REQUEST", "Prediction request is invalid.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise _error(500, "PREDICTION_FAILED", "Prediction could not be completed.") from exc
