from __future__ import annotations

from typing import Any

from apps.audit.models import AuditLog
from apps.core.request_context import get_request_id


def log_action(actor, action: str, target, payload: dict[str, Any] | None = None) -> AuditLog:
    return AuditLog.objects.create(
        actor=actor if getattr(actor, "is_authenticated", False) else None,
        action=action,
        target_type=target.__class__.__name__.lower()
        if not isinstance(target, str)
        else target.split(":")[0],
        target_id=str(target.pk) if not isinstance(target, str) else target.split(":", 1)[-1],
        payload=payload or {},
        request_id=get_request_id() or "",
    )
