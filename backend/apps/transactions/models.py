from django.conf import settings
from django.db import models

from apps.core.models import TimeStampedModel


class Transaction(TimeStampedModel):
    class Status(models.TextChoices):
        APPROVED = "approved", "Approved"
        REVIEW = "review", "Manual review"
        BLOCKED = "blocked", "Blocked"
        CONFIRMED_FRAUD = "confirmed_fraud", "Confirmed fraud"
        CLEARED = "cleared", "Cleared (legitimate)"

    transaction_id = models.BigIntegerField(unique=True)
    card1 = models.IntegerField(db_index=True)
    amount = models.DecimalField(max_digits=14, decimal_places=2)
    product_cd = models.CharField(max_length=4, blank=True)
    p_emaildomain = models.CharField(max_length=64, blank=True)
    device_type = models.CharField(max_length=32, blank=True)
    transaction_dt = models.BigIntegerField()
    payload = models.JSONField(default=dict)

    probability = models.FloatField(null=True)
    grade = models.CharField(max_length=16, blank=True)
    decision = models.CharField(max_length=16, blank=True)
    reasons = models.JSONField(default=list)
    model_version = models.CharField(max_length=32, blank=True)
    latency_ms = models.FloatField(null=True)
    status = models.CharField(
        max_length=20, choices=Status.choices, default=Status.APPROVED, db_index=True
    )
    is_flagged = models.BooleanField(default=False, db_index=True)
    reviewed_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL
    )
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"tx {self.transaction_id} {self.amount} {self.status}"
