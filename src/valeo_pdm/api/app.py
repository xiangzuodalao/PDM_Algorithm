from __future__ import annotations
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import uvicorn
from fastapi import FastAPI
from valeo_pdm.api.router import informer_router
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles
# app = FastAPI()


app = FastAPI(docs_url = None)
# 挂载本地静态文件目录
app.mount("/statics", StaticFiles(directory="statics"), name="statics")
@app.get("/docs",include_in_schema = False)
async def custom_docs():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        swagger_js_url = '/statics/swagger-ui/swagger-ui-bundle.js',
        swagger_css_url = '/statics/swagger-ui/swagger-ui.css'
    )
app.include_router(informer_router)

@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "10011"))
    uvicorn.run("valeo_pdm.api.app:app", host=host, port=port)
