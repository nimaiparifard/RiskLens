from django.db import models

from apps.core.models import TimeStampedModel


class DriftReport(TimeStampedModel):
    class Status(models.TextChoices):
        OK = "ok", "No significant drift"
        DRIFT = "drift", "Drift detected"
        SKIPPED = "skipped", "Not enough data"
        FAILED = "failed", "Failed"

    task = models.CharField(max_length=16, db_index=True)  # credit | fraud
    model_version = models.CharField(max_length=32, blank=True)
    window_start = models.DateTimeField()
    window_end = models.DateTimeField()
    n_samples = models.IntegerField(default=0)
    feature_psi = models.JSONField(default=dict)
    prediction_psi = models.FloatField(null=True, blank=True)
    max_psi = models.FloatField(null=True, blank=True)
    drifted_features = models.JSONField(default=list)
    status = models.CharField(max_length=16, choices=Status.choices, db_index=True)
    report_file = models.FileField(upload_to="drift_reports/", null=True, blank=True)
    retrain_triggered = models.BooleanField(default=False)
    error = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.task} {self.window_end:%Y-%m-%d} max_psi={self.max_psi}"
