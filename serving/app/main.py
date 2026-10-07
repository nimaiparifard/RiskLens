"""FastAPI app factory for the RiskLens model server."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api.routes import router
from app.config import Settings, get_settings
from app.feature_service import CreditFeatureService, FraudFeatureService
from app.history import InMemoryHistoryStore, RedisHistoryStore
from app.logs import RequestIdMiddleware, configure_logging
from app.metrics import MODEL_LOADED, PrometheusMiddleware, metrics_response
from app.model_loader import TASKS, ModelNotLoadedError, ModelRegistry

log = structlog.get_logger(__name__)


async def _poll_registry(app: FastAPI, interval: int) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            changed = await run_in_threadpool(app.state.registry.check_for_updates)
            if changed:
                log.info("models_hot_reloaded", changed=changed)
                _update_gauges(app.state.registry)
        except Exception as exc:  # keep polling
            log.warning("registry_poll_failed", error=str(exc))


def _update_gauges(registry: ModelRegistry) -> None:
    MODEL_LOADED.clear()
    for task, m in registry.loaded().items():
        MODEL_LOADED.labels(task, m.version).set(1)


def create_app(
    settings: Settings | None = None,
    registry: ModelRegistry | None = None,
    load_models: bool = True,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if load_models:
            versions = await run_in_threadpool(app.state.registry.load_all)
            log.info("startup_models", versions=versions)
        _update_gauges(app.state.registry)
        poller = None
        if settings.model_source == "mlflow" and settings.reload_interval_seconds > 0:
            poller = asyncio.create_task(_poll_registry(app, settings.reload_interval_seconds))
        yield
        if poller:
            poller.cancel()

    app = FastAPI(
        title="RiskLens Serving",
        version="0.1.0",
        description="Internal low-latency credit-risk and fraud scoring API",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.registry = registry or ModelRegistry(settings)
    store = (
        RedisHistoryStore(settings.redis_url, settings.history_max_events)
        if settings.history_backend == "redis"
        else InMemoryHistoryStore(settings.history_max_events)
    )
    app.state.credit_service = CreditFeatureService()
    app.state.fraud_service = FraudFeatureService(store)

    app.add_middleware(PrometheusMiddleware)
    app.add_middleware(RequestIdMiddleware)
    app.include_router(router)

    @app.exception_handler(ModelNotLoadedError)
    async def _not_loaded(_: Request, exc: ModelNotLoadedError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content={"error": "model_not_loaded", "task": exc.task, "detail": str(exc)},
        )

    @app.get("/health")
    def health() -> dict:
        loaded = app.state.registry.loaded()
        return {
            "status": "ok",
            "models": {t: (loaded[t].version if t in loaded else None) for t in TASKS},
        }

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        loaded = app.state.registry.loaded()
        ok = bool(loaded)
        return JSONResponse(
            status_code=200 if ok else 503, content={"ready": ok, "models": sorted(loaded)}
        )

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return metrics_response()

    return app


# run with:  uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8001
