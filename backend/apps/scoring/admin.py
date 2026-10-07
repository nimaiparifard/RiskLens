from django.contrib import admin

from apps.scoring.models import Prediction


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "created_at",
        "application",
        "model_version",
        "probability",
        "grade",
        "decision",
        "latency_ms",
    )
    list_filter = ("grade", "decision", "model_version")
    readonly_fields = [f.name for f in Prediction._meta.fields]
