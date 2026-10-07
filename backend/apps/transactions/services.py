from __future__ import annotations

from typing import Any

from django.db import transaction as db_transaction
from django.utils import timezone

from apps.audit.models import AuditLog
from apps.audit.services import log_action
from apps.core.exceptions import ValidationFailed
from apps.scoring.clients import get_serving_client
from apps.transactions.models import Transaction

DECISION_TO_STATUS = {
    "approve": Transaction.Status.APPROVED,
    "review": Transaction.Status.REVIEW,
    "block": Transaction.Status.BLOCKED,
}


def score_transaction(payload: dict[str, Any]) -> Transaction:
    """Real-time fraud check: serving decides approve / review / block."""
    result = get_serving_client().score_fraud(payload)  # reasons only for review/block
    with db_transaction.atomic():
        tx, _ = Transaction.objects.update_or_create(
            transaction_id=payload["TransactionID"],
            defaults={
                "card1": payload["card1"],
                "amount": payload["TransactionAmt"],
                "product_cd": payload.get("ProductCD") or "",
                "p_emaildomain": payload.get("P_emaildomain") or "",
                "device_type": payload.get("DeviceType") or "",
                "transaction_dt": payload["TransactionDT"],
                "payload": payload,
                "probability": result["probability"],
                "grade": result["grade"],
                "decision": result["decision"],
                "reasons": result.get("reasons", []),
                "model_version": str(result["model_version"]),
                "latency_ms": result.get("latency_ms"),
                "status": DECISION_TO_STATUS.get(result["decision"], Transaction.Status.REVIEW),
                "is_flagged": result["decision"] != "approve",
            },
        )
    return tx


def review_transaction(tx: Transaction, user, label: str, comment: str = "") -> Transaction:
    """Analyst label = ground truth for monitoring / retraining."""
    if label not in ("fraud", "legit"):
        raise ValidationFailed("label must be 'fraud' or 'legit'")
    tx.status = (
        Transaction.Status.CONFIRMED_FRAUD if label == "fraud" else Transaction.Status.CLEARED
    )
    tx.reviewed_by = user
    tx.reviewed_at = timezone.now()
    tx.save(update_fields=["status", "reviewed_by", "reviewed_at", "updated_at"])
    log_action(
        user,
        AuditLog.Action.TRANSACTION_REVIEWED,
        tx,
        {"label": label, "comment": comment, "model_decision": tx.decision},
    )
    return tx
