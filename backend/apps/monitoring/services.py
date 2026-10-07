"""Nightly drift report: last N hours of production inputs vs. the training reference.

PSI is computed by the serving service (it owns the reference sample and the
feature code). If ``risklens[monitoring]`` is installed in the worker, an
Evidently HTML report is rendered too and stored in MinIO / S3.
"""

from __future__ import annotations

from datetime import timedelta

import httpx
import structlog
from django.conf import settings
from django.core.files.base import ContentFile
from django.utils import timezone

from apps.audit.models import AuditLog
from apps.audit.services import log_action
from apps.core.exceptions import DomainError
from apps.monitoring.models import DriftReport
from apps.scoring.clients import get_serving_client
from apps.scoring.models import Prediction
from apps.transactions.models import Transaction

log = structlog.get_logger(__name__)


def _window_data(task: str, start, end) -> tuple[list[dict], list[float]]:
    if task == "credit":
        rows = Prediction.objects.filter(created_at__gte=start, created_at__lt=end).values_list(
            "input_payload", "probability"
        )
    else:
        rows = Transaction.objects.filter(created_at__gte=start, created_at__lt=end).values_list(
            "payload", "probability"
        )
    rows = [(p, s) for p, s in rows if p and s is not None]
    return [p for p, _ in rows], [s for _, s in rows]


def run_drift_report(task: str = "credit", hours: int | None = None) -> DriftReport:
    end = timezone.now()
    start = end - timedelta(hours=hours or settings.DRIFT_WINDOW_HOURS)
    records, scores = _window_data(task, start, end)
    report = DriftReport(task=task, window_start=start, window_end=end, n_samples=len(records))
    if len(records) < settings.DRIFT_MIN_SAMPLES:
        report.status = DriftReport.Status.SKIPPED
        report.save()
        return report
    client = get_serving_client()
    try:
        result = client.drift(task, records, scores, settings.DRIFT_PSI_THRESHOLD)
    except DomainError as exc:
        report.status = DriftReport.Status.FAILED
        report.error = exc.message
        report.save()
        return report

    report.model_version = str(result.get("model_version", ""))
    report.feature_psi = result["feature_psi"]
    report.prediction_psi = result.get("prediction_psi")
    report.max_psi = result["max_psi"]
    report.drifted_features = result.get("drifted_features", [])
    report.status = DriftReport.Status.DRIFT if result["drift_detected"] else DriftReport.Status.OK
    report.save()
    _attach_evidently(report, client, task, records)

    if report.status == DriftReport.Status.DRIFT:
        log_action(
            None,
            AuditLog.Action.DRIFT_DETECTED,
            report,
            {
                "task": task,
                "max_psi": report.max_psi,
                "features": report.drifted_features[:10],
            },
        )
        log.warning(
            "drift_detected",
            task=task,
            max_psi=report.max_psi,
            features=report.drifted_features[:5],
        )
        if settings.AUTO_RETRAIN_ON_DRIFT:
            report.retrain_triggered = trigger_retrain(
                f"drift detected for {task} (max PSI {report.max_psi:.3f})"
            )
            report.save(update_fields=["retrain_triggered"])
    return report


def _attach_evidently(report: DriftReport, client, task: str, records: list[dict]) -> None:
    try:
        import tempfile
        from pathlib import Path

        import pandas as pd

        from risklens.monitoring.drift import evidently_report
        from risklens.scorer import prepare_input
    except ImportError:
        return  # optional dependency
    try:
        reference = pd.DataFrame(client.reference(task)["records"]).drop(
            columns=["score"], errors="ignore"
        )
        current = pd.DataFrame.from_records(records)
        if task == "credit":
            current = prepare_input("credit", current)
        with tempfile.TemporaryDirectory() as tmp:
            path = evidently_report(
                reference,
                current[[c for c in reference.columns if c in current]],
                Path(tmp) / "r.html",
            )
            if path:
                report.report_file.save(
                    f"{task}-{report.window_end:%Y%m%d-%H%M}.html", ContentFile(path.read_bytes())
                )
    except Exception as exc:  # the PSI numbers are already stored; HTML is a bonus
        log.warning("evidently_report_failed", error=str(exc))


def trigger_retrain(reason: str) -> bool:
    """Dispatch the GitHub Actions ml-train workflow (needs GITHUB_TOKEN + GITHUB_REPOSITORY)."""
    if not (settings.GITHUB_TOKEN and settings.GITHUB_REPOSITORY):
        log.info("retrain_not_configured", reason=reason)
        return False
    url = f"https://api.github.com/repos/{settings.GITHUB_REPOSITORY}/actions/workflows/ml-train.yml/dispatches"
    r = httpx.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.GITHUB_TOKEN}",
            "Accept": "application/vnd.github+json",
        },
        json={"ref": "main", "inputs": {"reason": reason}},
        timeout=10,
    )
    ok = r.status_code == 204
    log_action(
        None, AuditLog.Action.RETRAIN_TRIGGERED, "workflow:ml-train", {"reason": reason, "ok": ok}
    )
    return ok
