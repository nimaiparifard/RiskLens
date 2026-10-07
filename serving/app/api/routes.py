"""Internal scoring API (called by the Django backend, never by browsers).

POST /v1/credit/score        single application -> PD, grade, decision, top-5 reasons
POST /v1/credit/score/batch  many applications (Celery batch scoring)
POST /v1/fraud/score         single transaction (< 50 ms budget)
GET  /v1/model/info          versions, thresholds, metrics
POST /v1/model/reload        hot-reload from the registry (internal token)
GET  /v1/model/reference     reference sample for drift monitoring
POST /v1/monitoring/drift    PSI of recent inputs vs the training reference
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import pandas as pd
import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.deps import (
    get_credit_service,
    get_fraud_service,
    get_registry,
    require_internal_token,
)
from app.feature_service import CreditFeatureService, FraudFeatureService
from app.metrics import ERRORS, SHADOW_DIFF, observe_prediction
from app.model_loader import TASKS, LoadedModel, ModelRegistry
from app.schemas import (
    BatchCreditRequest,
    BatchResponse,
    CreditScoreRequest,
    DriftRequest,
    FraudScoreRequest,
    ModelInfo,
    ReasonOut,
    ScoreResponse,
    ShadowResult,
)
from risklens.interfaces import ScoreResult

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/v1")


def _response(
    task: str,
    model: LoadedModel,
    res: ScoreResult,
    seconds: float,
    request: Request,
    shadow: ShadowResult | None = None,
) -> ScoreResponse:
    return ScoreResponse(
        task=task,  # type: ignore[arg-type]
        probability=round(res.probability, 6),
        score_points=res.points,
        grade=res.grade or "",
        decision=res.decision,
        threshold=round(res.threshold, 6),
        reasons=[ReasonOut(**dataclasses.asdict(r)) for r in res.reasons],
        model_name=model.scorer.metadata.get("config", {}).get("registered_model", task),
        model_version=model.version,
        latency_ms=round(seconds * 1000, 3),
        request_id=getattr(request.state, "request_id", None),
        shadow=shadow,
    )


def _shadow(
    registry: ModelRegistry, task: str, score_fn, champion: ScoreResult
) -> ShadowResult | None:
    """Score with the challenger too (shadow mode). Never affects the decision."""
    chall = registry.challenger(task)
    if chall is None:
        return None
    try:
        res = score_fn(chall.scorer)
    except Exception as exc:  # shadow failures must never break production traffic
        ERRORS.labels(task, "shadow").inc()
        log.warning("shadow_failed", task=task, error=str(exc))
        return None
    SHADOW_DIFF.labels(task).observe(abs(res.probability - champion.probability))
    log.info(
        "shadow_score",
        task=task,
        champion=champion.probability,
        challenger=res.probability,
        challenger_version=chall.version,
    )
    return ShadowResult(
        model_version=chall.version, probability=round(res.probability, 6), decision=res.decision
    )


@router.post("/credit/score", response_model=ScoreResponse)
def score_credit(
    body: CreditScoreRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    service: CreditFeatureService = Depends(get_credit_service),
) -> ScoreResponse:
    model = registry.get("credit")
    features = body.features.model_dump()
    t0 = time.perf_counter()
    try:
        res = service.score(model.scorer, features, explain=body.explain)
    except Exception as exc:
        ERRORS.labels("credit", type(exc).__name__).inc()
        log.exception("credit_scoring_failed")
        raise HTTPException(422, f"could not score application: {exc}") from exc
    seconds = time.perf_counter() - t0
    observe_prediction("credit", model.version, res.probability, res.decision, seconds)
    shadow = _shadow(registry, "credit", lambda s: service.score(s, features, explain=False), res)
    log.info(
        "credit_scored",
        model_version=model.version,
        probability=res.probability,
        decision=res.decision,
        latency_ms=round(seconds * 1000, 2),
    )
    return _response("credit", model, res, seconds, request, shadow)


@router.post("/credit/score/batch", response_model=BatchResponse)
def score_credit_batch(
    body: BatchCreditRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    service: CreditFeatureService = Depends(get_credit_service),
) -> BatchResponse:
    model = registry.get("credit")
    t0 = time.perf_counter()
    results = service.score_batch(
        model.scorer, [i.model_dump() for i in body.items], explain=body.explain
    )
    seconds = time.perf_counter() - t0
    per_item = seconds / len(results)
    for r in results:
        observe_prediction("credit", model.version, r.probability, r.decision, per_item)
    return BatchResponse(
        results=[_response("credit", model, r, per_item, request) for r in results]
    )


@router.post("/fraud/score", response_model=ScoreResponse)
def score_fraud(
    body: FraudScoreRequest,
    request: Request,
    registry: ModelRegistry = Depends(get_registry),
    service: FraudFeatureService = Depends(get_fraud_service),
) -> ScoreResponse:
    model = registry.get("fraud")
    tx = body.transaction.model_dump()
    t0 = time.perf_counter()
    try:
        res = service.score(model.scorer, tx, explain=body.explain, record=body.record_history)
    except Exception as exc:
        ERRORS.labels("fraud", type(exc).__name__).inc()
        log.exception("fraud_scoring_failed")
        raise HTTPException(422, f"could not score transaction: {exc}") from exc
    seconds = time.perf_counter() - t0
    observe_prediction("fraud", model.version, res.probability, res.decision, seconds)
    shadow = _shadow(
        registry, "fraud", lambda s: service.score(s, tx, explain=False, record=False), res
    )
    log.info(
        "fraud_scored",
        model_version=model.version,
        probability=res.probability,
        decision=res.decision,
        latency_ms=round(seconds * 1000, 2),
    )
    return _response("fraud", model, res, seconds, request, shadow)


@router.get("/model/info", response_model=dict[str, ModelInfo])
def model_info(registry: ModelRegistry = Depends(get_registry)) -> dict[str, ModelInfo]:
    out = {}
    loaded = registry.loaded()
    for task in TASKS:
        m = loaded.get(task)
        if m is None:
            out[task] = ModelInfo(task=task, loaded=False, model_name=registry.model_name(task))
            continue
        info = m.scorer.info()
        out[task] = ModelInfo(
            task=task,
            loaded=True,
            model_name=registry.model_name(task),
            version=m.version,
            source=m.source,
            loaded_at=m.loaded_at,
            threshold=info["threshold"],
            metrics=info["metrics"],
            grade_bands=info["grade_bands"],
            n_features=info["n_features"],
            model=info["model"],
            trained_at=info["trained_at"],
            git_sha=info["git_sha"],
            challenger_version=(c.version if (c := registry.challenger(task)) else None),
        )
    return out


@router.post("/model/reload", dependencies=[Depends(require_internal_token)])
def reload_models(
    task: str | None = None, registry: ModelRegistry = Depends(get_registry)
) -> dict[str, str]:
    tasks = [task] if task else list(TASKS)
    return {t: registry.reload(t) for t in tasks}


@router.get("/model/reference")
def reference(
    task: str = Query("credit", pattern="^(credit|fraud)$"),
    n: int = Query(2000, ge=10, le=5000),
    registry: ModelRegistry = Depends(get_registry),
) -> dict[str, Any]:
    model = registry.get(task)
    return {
        "task": task,
        "model_version": model.version,
        "records": model.scorer.reference_records(n),
    }


@router.post("/monitoring/drift")
def drift(body: DriftRequest, registry: ModelRegistry = Depends(get_registry)) -> dict[str, Any]:
    """PSI per feature of recent production inputs vs. the model's training reference."""
    from risklens.monitoring.drift import compute_drift

    model = registry.get(body.task)
    ref = model.scorer.reference
    if ref is None:
        raise HTTPException(409, "model has no reference sample")
    current = (
        model.scorer.prepare(pd.DataFrame.from_records(body.records))
        if body.task == "credit"
        else pd.DataFrame.from_records(body.records)
    )
    if body.scores is not None:
        current = current.assign(score=body.scores[: len(current)])
    result = compute_drift(
        ref,
        current,
        features=[c for c in ref.columns if c != "score"],
        threshold=body.psi_threshold,
    )
    return {"model_version": model.version, **result.to_dict()}
