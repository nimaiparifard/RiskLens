from __future__ import annotations

from typing import Any

from django.db import transaction

from apps.applicants.models import Applicant, Application
from apps.audit.models import AuditLog
from apps.audit.services import log_action


@transaction.atomic
def create_application(
    *,
    applicant: dict[str, Any],
    features: dict[str, Any],
    history: dict[str, Any] | None = None,
    user=None,
) -> Application:
    person = None
    if applicant.get("external_id"):
        person = Applicant.objects.filter(external_id=applicant["external_id"]).first()
    if person is None:
        person = Applicant.objects.create(**applicant)
    app = Application.objects.create(
        applicant=person,
        contract_type=features.get("NAME_CONTRACT_TYPE") or "Cash loans",
        amt_credit=features["AMT_CREDIT"],
        amt_annuity=features.get("AMT_ANNUITY"),
        amt_income_total=features["AMT_INCOME_TOTAL"],
        features=features,
        history=history or {},
        created_by=user if getattr(user, "is_authenticated", False) else None,
    )
    log_action(user, AuditLog.Action.APPLICATION_CREATED, app, {"reference": app.reference})
    return app
