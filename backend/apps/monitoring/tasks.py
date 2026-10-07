from celery import shared_task

from apps.monitoring import services


@shared_task
def run_drift_report(task: str = "credit") -> dict:
    report = services.run_drift_report(task)
    return {"id": report.pk, "status": report.status, "max_psi": report.max_psi}


@shared_task
def trigger_retrain(reason: str = "manual") -> bool:
    return services.trigger_retrain(reason)
