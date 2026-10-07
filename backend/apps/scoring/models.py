from django.db import models

from apps.core.models import UUIDModel


class Prediction(UUIDModel):
    """One model score of one application, stored with its explanation (audit-ready)."""

    application = models.ForeignKey(
        "applicants.Application", related_name="predictions", on_delete=models.CASCADE
    )
    model_name = models.CharField(max_length=64)
    model_version = models.CharField(max_length=32, db_index=True)
    probability = models.FloatField(help_text="calibrated probability of default")
    score_points = models.IntegerField(null=True, blank=True)
    grade = models.CharField(max_length=8, db_index=True)
    decision = models.CharField(
        max_length=16, help_text="model recommendation: approve / review / reject"
    )
    threshold = models.FloatField()
    shap_reasons = models.JSONField(default=list)
    latency_ms = models.FloatField()
    input_payload = models.JSONField(
        default=dict, help_text="exact features sent (drift monitoring)"
    )
    shadow = models.JSONField(null=True, blank=True, help_text="challenger score in shadow mode")
    requested_by = models.ForeignKey("users.User", null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["model_name", "created_at"])]

    def __str__(self) -> str:
        return f"{self.application_id} pd={self.probability:.3f} {self.grade} v{self.model_version}"
