"""Domain exceptions mapped to HTTP errors (Rule 8: fail loudly)."""

from __future__ import annotations

import structlog
from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import exception_handler

log = structlog.get_logger(__name__)


class DomainError(Exception):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "domain_error"

    def __init__(self, message: str, **extra: object) -> None:
        super().__init__(message)
        self.message = message
        self.extra = extra


class InvalidStateError(DomainError):
    status_code = status.HTTP_409_CONFLICT
    code = "invalid_state"


class ValidationFailed(DomainError):
    code = "validation_failed"


class PermissionDeniedError(DomainError):
    status_code = status.HTTP_403_FORBIDDEN
    code = "permission_denied"


class ServingUnavailable(DomainError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "serving_unavailable"


class ServingRejected(DomainError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "serving_rejected"


class ExternalServiceError(DomainError):
    status_code = status.HTTP_502_BAD_GATEWAY
    code = "external_service_error"


def api_exception_handler(exc, context):
    if isinstance(exc, DomainError):
        log.warning(
            "domain_error",
            code=exc.code,
            message=exc.message,
            **{k: str(v) for k, v in exc.extra.items()},
        )
        return Response(
            {"error": exc.code, "detail": exc.message, **exc.extra}, status=exc.status_code
        )
    response = exception_handler(exc, context)
    if response is not None and isinstance(response.data, dict) and "error" not in response.data:
        response.data = {"error": getattr(exc, "default_code", "error"), **response.data}
    return response
