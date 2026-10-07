"""WRITE logic for scoring and officer decisions (Rule 4: thin views, logic here)."""

from __future__ import annotations

import structlog
from django.db import transaction
from django.utils import timezone

from apps.applicants.models import Application
from apps.audit.models import AuditLog
from apps.audit.services import log_action
from apps.core.exceptions import InvalidStateError, PermissionDeniedError, ValidationFailed
from apps.scoring.clients import get_serving_client
from apps.scoring.models import Prediction

log = structlog.get_logger(__name__)

FINAL_STATES = {Application.Status.APPROVED, Application.Status.REJECTED}


def _model_payload(application: Application) -> dict:
    return {**(application.features or {}), **(application.history or {})}


def _store_prediction(
    application: Application, result: dict, payload: dict, user=None
) -> Prediction:
    pred = Prediction.objects.create(
        application=application,
        model_name=result.get("model_name", "risklens-credit"),
        model_version=str(result["model_version"]),
        probability=result["probability"],
        score_points=result.get("score_points"),
        grade=result["grade"],
        decision=result["decision"],
        threshold=result["threshold"],
        shap_reasons=result.get("reasons", []),
        latency_ms=result.get("latency_ms", 0.0),
        input_payload=payload,
        shadow=result.get("shadow"),
        requested_by=user if getattr(user, "is_authenticated", False) else None,
    )
    application.latest_probability = pred.probability
    application.latest_grade = pred.grade
    application.recommendation = pred.decision
    if application.status == Application.Status.PENDING:
        application.status = Application.Status.SCORED
    application.save(
        update_fields=[
            "latest_probability",
            "latest_grade",
            "recommendation",
            "status",
            "updated_at",
        ]
    )
    return pred


@transaction.atomic
def score_application(application: Application, user=None) -> Prediction:
    """Call the serving API, persist the prediction with its SHAP reasons, audit it."""
    if application.status in FINAL_STATES:
        raise InvalidStateError(
            "application already decided; re-scoring is not allowed", status=application.status
        )
    payload = _model_payload(application)
    result = get_serving_client().score_credit(payload, explain=True)
    pred = _store_prediction(application, result, payload, user)
    log_action(
        user,
        AuditLog.Action.APPLICATION_SCORED,
        application,
        {
            "prediction_id": str(pred.id),
            "probability": pred.probability,
            "grade": pred.grade,
            "recommendation": pred.decision,
            "model_version": pred.model_version,
        },
    )
    log.info(
        "application_scored",
        application_id=str(application.id),
        model_version=pred.model_version,
        probability=pred.probability,
        grade=pred.grade,
    )
    return pred


def batch_score(applications: list[Application]) -> int:
    """Score many pending applications with one serving call (Celery)."""
    if not applications:
        return 0
    payloads = [_model_payload(a) for a in applications]
    results = get_serving_client().score_credit_batch(payloads)
    with transaction.atomic():
        for app, payload, result in zip(applications, payloads, results, strict=True):
            pred = _store_prediction(app, result, payload)
            log_action(
                None,
                AuditLog.Action.APPLICATION_SCORED,
                app,
                {
                    "prediction_id": str(pred.id),
                    "probability": pred.probability,
                    "grade": pred.grade,
                    "recommendation": pred.decision,
                    "model_version": pred.model_version,
                    "batch": True,
                },
            )
    return len(results)


@transaction.atomic
def decide(application: Application, user, decision: str, comment: str = "") -> Application:
    """Officer approves / rejects.

    * a decision AGAINST the model recommendation is an override -> comment required
    * changing an already-final decision -> admin only (override_decision)
    """
    if decision not in ("approve", "reject"):
        raise ValidationFailed("decision must be 'approve' or 'reject'")
    if not getattr(user, "is_officer", False):
        raise PermissionDeniedError("only credit officers can decide")
    latest = application.predictions.order_by("-created_at").first()
    if latest is None:
        raise InvalidStateError("score the application before deciding")
    if application.status in FINAL_STATES:
        return override_decision(application, user, decision, comment)

    is_override = (
        latest.decision != decision
    )  # 'review' recommendations always need a justification
    if is_override and not comment.strip():
        raise ValidationFailed(
            "a comment is required when the decision differs from the model recommendation",
            recommendation=latest.decision,
        )
    _apply(application, user, decision, comment, is_override)
    log_action(
        user,
        AuditLog.Action.DECISION_MADE,
        application,
        {
            "decision": decision,
            "recommendation": latest.decision,
            "is_override": is_override,
            "comment": comment,
            "prediction_id": str(latest.id),
            "model_version": latest.model_version,
        },
    )
    return application


def override_decision(application: Application, user, decision: str, comment: str) -> Application:
    if getattr(user, "role", None) != "admin":
        raise PermissionDeniedError("changing a final decision requires the admin role")
    if not comment.strip():
        raise ValidationFailed("a comment is required to override a final decision")
    previous = application.status
    _apply(application, user, decision, comment, True)
    log_action(
        user,
        AuditLog.Action.DECISION_OVERRIDDEN,
        application,
        {
            "previous_status": previous,
            "decision": decision,
            "comment": comment,
        },
    )
    return application


def _apply(application: Application, user, decision: str, comment: str, is_override: bool) -> None:
    application.status = (
        Application.Status.APPROVED if decision == "approve" else Application.Status.REJECTED
    )
    application.decided_by = user
    application.decided_at = timezone.now()
    application.decision_comment = comment
    application.is_override = is_override
    application.save(
        update_fields=[
            "status",
            "decided_by",
            "decided_at",
            "decision_comment",
            "is_override",
            "updated_at",
        ]
    )
