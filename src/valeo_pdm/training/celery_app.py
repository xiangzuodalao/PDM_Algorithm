from __future__ import annotations

import os

from celery import Celery


def _visibility_timeout() -> int:
    raw = os.getenv("VALEO_PDM_CELERY_VISIBILITY_TIMEOUT", "86400")
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError("VALEO_PDM_CELERY_VISIBILITY_TIMEOUT 必须为正整数") from exc
    if value <= 0:
        raise RuntimeError("VALEO_PDM_CELERY_VISIBILITY_TIMEOUT 必须为正整数")
    return value


app = Celery(
    "valeo_pdm",
    broker=os.getenv("VALEO_PDM_CELERY_BROKER_URL", "redis://127.0.0.1:6379/0"),
    include=["valeo_pdm.training.tasks"],
)

visibility_timeout = _visibility_timeout()
app.conf.update(
    accept_content=["json"],
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": visibility_timeout},
    result_backend=None,
    task_acks_late=True,
    task_default_queue="training",
    task_ignore_result=True,
    task_serializer="json",
    visibility_timeout=visibility_timeout,
    worker_prefetch_multiplier=1,
)
