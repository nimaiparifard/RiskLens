"""READ logic: querysets and dashboard aggregates."""

from __future__ import annotations

from django.db.models import Avg, Count, Q, QuerySet

from apps.applicants.models import Application


def application_list() -> QuerySet[Application]:
    return Application.objects.select_related("applicant", "decided_by")


def application_stats() -> dict:
    qs = Application.objects.all()
    by_status = dict(qs.values_list("status").annotate(n=Count("id")))
    by_grade = dict(qs.exclude(latest_grade="").values_list("latest_grade").annotate(n=Count("id")))
    decided = qs.filter(status__in=[Application.Status.APPROVED, Application.Status.REJECTED])
    agg = decided.aggregate(
        approved=Count("id", filter=Q(status=Application.Status.APPROVED)),
        overrides=Count("id", filter=Q(is_override=True)),
        total=Count("id"),
    )
    return {
        "total": qs.count(),
        "by_status": by_status,
        "by_grade": by_grade,
        "approval_rate": (agg["approved"] / agg["total"]) if agg["total"] else None,
        "override_rate": (agg["overrides"] / agg["total"]) if agg["total"] else None,
        "avg_probability": qs.aggregate(v=Avg("latest_probability"))["v"],
    }
