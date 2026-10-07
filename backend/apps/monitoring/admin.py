from django.contrib import admin

from apps.monitoring.models import DriftReport


@admin.register(DriftReport)
class DriftReportAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "task",
        "model_version",
        "n_samples",
        "max_psi",
        "status",
        "retrain_triggered",
    )
    list_filter = ("task", "status")
