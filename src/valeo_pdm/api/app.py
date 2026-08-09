from __future__ import annotations

from contextlib import asynccontextmanager
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from valeo_pdm.api.router import informer_router
from valeo_pdm.api.prediction_v2 import router as prediction_v2_router
from valeo_pdm.prediction_v2.catalog import (
    ModelCatalog,
    isolated_fixture_mode_enabled,
    manifest_is_isolated_pilot,
    validate_isolated_runtime_environment,
)
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles


@asynccontextmanager
async def prediction_runtime_lifespan(_: FastAPI):
    try:
        validate_isolated_runtime_environment()
        manifest_value = os.getenv("VALEO_PDM_PREDICTION_V2_MANIFEST")
        if manifest_value and manifest_is_isolated_pilot(Path(manifest_value)):
            if not isolated_fixture_mode_enabled():
                raise ValueError("isolated pilot fixture manifest requires isolated fixture mode")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(docs_url=None, lifespan=prediction_runtime_lifespan)
# 挂载本地静态文件目录
app.mount("/statics", StaticFiles(directory="statics"), name="statics")


@app.get("/docs", include_in_schema=False)
async def custom_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url="/statics/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/statics/swagger-ui/swagger-ui.css",
    )


app.include_router(informer_router)
app.include_router(prediction_v2_router)


@app.exception_handler(RequestValidationError)
async def prediction_v2_validation_error(request: Request, _exc: RequestValidationError):
    if request.url.path.startswith("/api/v2/"):
        return JSONResponse(
            status_code=422,
            content={
                "code": "INVALID_PREDICTION_REQUEST",
                "message": "Prediction request is invalid.",
            },
        )
    return JSONResponse(status_code=422, content={"detail": _exc.errors()})


@app.exception_handler(HTTPException)
async def prediction_v2_http_error(request: Request, exc: HTTPException):
    if request.url.path.startswith("/api/v2/") and isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail, headers=exc.headers)
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail}, headers=exc.headers
    )


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
def readyz() -> JSONResponse:
    try:
        validate_isolated_runtime_environment()
        manifest_value = os.getenv("VALEO_PDM_PREDICTION_V2_MANIFEST")
        object_root_value = os.getenv("VALEO_PDM_PREDICTION_V2_OBJECT_ROOT")
        if not manifest_value or not object_root_value:
            raise ValueError("prediction runtime is not configured")
        catalog = ModelCatalog.from_manifest(
            Path(manifest_value),
            object_root=Path(object_root_value),
            allowed_tenant_ids=os.getenv("VALEO_PDM_ALLOWED_TENANT_IDS", "").split(","),
        )
        if not catalog.readiness():
            raise ValueError("prediction runtime is not ready")
    except (OSError, ValueError):
        return JSONResponse(status_code=503, content={"status": "not_ready"})
    return JSONResponse(status_code=200, content={"status": "ok"})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10011"))
    uvicorn.run("valeo_pdm.api.app:app", host=host, port=port)
