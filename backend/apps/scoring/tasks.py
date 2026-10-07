"""Celery tasks: thin wrappers around services."""

import structlog
from celery import shared_task

from apps.applicants.models import Application
from apps.scoring import services

log = structlog.get_logger(__name__)


@shared_task(autoretry_for=(Exception,), retry_backoff=True, max_retries=3)
def score_application_task(application_id: str) -> str:
    app = Application.objects.get(pk=application_id)
    return str(services.score_application(app).id)


@shared_task
def batch_score_pending(limit: int = 500) -> int:
    pending = list(
        Application.objects.filter(status=Application.Status.PENDING).order_by("created_at")[:limit]
    )
    n = services.batch_score(pending)
    log.info("batch_scored", count=n)
    return n
