from __future__ import annotations
import os
import uvicorn
from fastapi import FastAPI
from valeo_pdm.api.router import informer_router

app = FastAPI()
app.include_router(informer_router)


@app.get("/healthz", include_in_schema=False)
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("valeo_pdm.api.app:app", host=host, port=port)
