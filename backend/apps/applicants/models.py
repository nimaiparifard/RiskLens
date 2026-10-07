from django.conf import settings
from django.db import models

from apps.core.models import UUIDModel


class Applicant(UUIDModel):
    external_id = models.CharField(
        max_length=32, blank=True, db_index=True, help_text="e.g. Home Credit SK_ID_CURR"
    )
    full_name = models.CharField(max_length=200)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=32, blank=True)

    def __str__(self) -> str:
        return self.full_name


class Application(UUIDModel):
    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        SCORED = "scored", "Scored – awaiting decision"
        APPROVED = "approved", "Approved"
        REJECTED = "rejected", "Rejected"

    reference = models.CharField(max_length=20, unique=True, editable=False)
    applicant = models.ForeignKey(Applicant, related_name="applications", on_delete=models.PROTECT)
    contract_type = models.CharField(max_length=32, default="Cash loans")
    amt_credit = models.DecimalField(max_digits=14, decimal_places=2)
    amt_annuity = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    amt_income_total = models.DecimalField(max_digits=14, decimal_places=2)
    # model inputs (Home Credit field names) + pre-computed credit-history aggregates
    features = models.JSONField(default=dict)
    history = models.JSONField(default=dict, blank=True)

    status = models.CharField(
        max_length=16, choices=Status.choices, default=Status.PENDING, db_index=True
    )
    latest_probability = models.FloatField(null=True, blank=True)
    latest_grade = models.CharField(max_length=8, blank=True, db_index=True)
    recommendation = models.CharField(max_length=16, blank=True)

    decided_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="decisions",
    )
    decided_at = models.DateTimeField(null=True, blank=True)
    decision_comment = models.TextField(blank=True)
    is_override = models.BooleanField(default=False)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )

    class Meta:
        ordering = ["-created_at"]

    def save(self, *args, **kwargs):
        if not self.reference:
            last = (
                Application.objects.order_by("-reference")
                .values_list("reference", flat=True)
                .first()
            )
            n = int(last.split("-")[1]) + 1 if last else 1
            self.reference = f"APP-{n:06d}"
        super().save(*args, **kwargs)

    def __str__(self) -> str:
        return f"{self.reference} ({self.status})"
