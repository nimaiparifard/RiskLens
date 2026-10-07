from django.conf import settings
from django.db import models


class AuditLog(models.Model):
    """Append-only record of who did what (compliance: every officer decision / override)."""

    class Action(models.TextChoices):
        APPLICATION_CREATED = "application.created"
        APPLICATION_SCORED = "application.scored"
        DECISION_MADE = "decision.made"
        DECISION_OVERRIDDEN = "decision.overridden"
        TRANSACTION_REVIEWED = "transaction.reviewed"
        MODEL_RELOADED = "model.reloaded"
        DRIFT_DETECTED = "drift.detected"
        RETRAIN_TRIGGERED = "retrain.triggered"

    actor = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL
    )
    action = models.CharField(max_length=64, choices=Action.choices, db_index=True)
    target_type = models.CharField(max_length=64, db_index=True)
    target_id = models.CharField(max_length=64, db_index=True)
    payload = models.JSONField(default=dict, blank=True)
    request_id = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.created_at:%Y-%m-%d %H:%M} {self.actor} {self.action} {self.target_type}:{self.target_id}"

    def save(self, *args, **kwargs):
        if self.pk:
            raise ValueError("audit log entries are immutable")
        super().save(*args, **kwargs)
