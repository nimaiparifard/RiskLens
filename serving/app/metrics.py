"""Prometheus instrumentation: request count/latency, model latency,
prediction distribution and decisions - all labelled with model_version."""

from __future__ import annotations

import time

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

REQUESTS = Counter("risklens_http_requests_total", "HTTP requests", ["method", "route", "status"])
REQUEST_LATENCY = Histogram(
    "risklens_http_request_duration_seconds",
    "HTTP request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1, 2.5),
)
MODEL_LATENCY = Histogram(
    "risklens_model_latency_seconds",
    "features + model + SHAP latency",
    ["task", "model_version"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5),
)
PREDICTIONS = Histogram(
    "risklens_prediction_probability",
    "distribution of predicted probabilities",
    ["task", "model_version"],
    buckets=(0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0),
)
DECISIONS = Counter("risklens_decisions_total", "decisions", ["task", "decision", "model_version"])
ERRORS = Counter("risklens_errors_total", "errors by type", ["task", "type"])
MODEL_LOADED = Gauge("risklens_model_loaded", "1 if a model is loaded", ["task", "model_version"])
SHADOW_DIFF = Histogram(
    "risklens_shadow_abs_diff",
    "|champion - challenger| probability",
    ["task"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
)


def observe_prediction(
    task: str, version: str, probability: float, decision: str, seconds: float
) -> None:
    MODEL_LATENCY.labels(task, version).observe(seconds)
    PREDICTIONS.labels(task, version).observe(probability)
    DECISIONS.labels(task, decision, version).inc()


class PrometheusMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = getattr(route, "path", "unmatched")
            if path != "/metrics":
                REQUESTS.labels(request.method, path, str(status)).inc()
                REQUEST_LATENCY.labels(path).observe(time.perf_counter() - start)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
