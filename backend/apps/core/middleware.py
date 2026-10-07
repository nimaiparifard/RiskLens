"""Bind request_id (propagated to the serving service) and user_id to logs."""

import time
import uuid

import structlog

from apps.core.request_context import set_request_id

log = structlog.get_logger("http")
HEADER = "HTTP_X_REQUEST_ID"


class RequestContextMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request_id = request.META.get(HEADER) or uuid.uuid4().hex
        request.request_id = request_id
        set_request_id(request_id)
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id, service="backend")
        start = time.perf_counter()
        response = self.get_response(request)
        user = getattr(request, "user", None)
        if request.path.startswith("/api/") and request.path != "/api/health/":
            log.info(
                "request",
                method=request.method,
                path=request.path,
                status=response.status_code,
                user_id=getattr(user, "pk", None),
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
            )
        response["X-Request-ID"] = request_id
        return response
