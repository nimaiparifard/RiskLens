from __future__ import annotations

import time

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.model_loader import LoadedModel, ModelRegistry
from risklens.scorer import Scorer


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["models"]["credit"] is not None
    assert client.get("/health/ready").status_code == 200


def test_credit_score_response_schema(client, credit_payload):
    r = client.post("/v1/credit/score", json={"features": credit_payload})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1
    assert body["grade"] in {"A", "B", "C", "D", "E"}
    assert body["decision"] in {"approve", "review", "reject"}
    assert len(body["reasons"]) == 5
    assert {"feature", "value", "contribution", "direction", "text"} <= set(body["reasons"][0])
    assert isinstance(body["score_points"], int)
    assert r.headers["X-Request-ID"] == body["request_id"]


def test_credit_minimal_payload_and_extra_fields(client):
    payload = {
        "AMT_INCOME_TOTAL": 120000,
        "AMT_CREDIT": 400000,
        "DAYS_BIRTH": -12000,
        "EXT_SOURCE_2": 0.4,
        "BURO_COUNT": 3,
        "INSTAL_DPD_MEAN": 1.5,
    }
    r = client.post("/v1/credit/score", json={"features": payload, "explain": False})
    assert r.status_code == 200, r.text
    assert r.json()["reasons"] == []


def test_credit_validation_errors(client):
    bad = {"AMT_INCOME_TOTAL": -1, "AMT_CREDIT": 1000, "DAYS_BIRTH": -10000}
    assert client.post("/v1/credit/score", json={"features": bad}).status_code == 422
    missing = {"AMT_CREDIT": 1000}
    assert client.post("/v1/credit/score", json={"features": missing}).status_code == 422
    out_of_range = {"AMT_INCOME_TOTAL": 1, "AMT_CREDIT": 1, "DAYS_BIRTH": -1, "EXT_SOURCE_1": 3}
    assert client.post("/v1/credit/score", json={"features": out_of_range}).status_code == 422


def test_credit_batch(client, credit_payload):
    r = client.post("/v1/credit/score/batch", json={"items": [credit_payload] * 3})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert len({round(x["probability"], 8) for x in results}) == 1


def test_fraud_score_and_history(client, fraud_payload, settings):
    first = client.post("/v1/fraud/score", json={"transaction": fraud_payload, "explain": True})
    assert first.status_code == 200, first.text
    body = first.json()
    assert body["decision"] in {"approve", "review", "block"}
    assert len(body["reasons"]) == 3
    store = client.app.state.fraud_service.store
    assert len(store.get(str(fraud_payload["card1"]))) == 1
    nxt = dict(
        fraud_payload,
        TransactionID=fraud_payload["TransactionID"] + 1,
        TransactionDT=fraud_payload["TransactionDT"] + 60,
    )
    assert client.post("/v1/fraud/score", json={"transaction": nxt}).status_code == 200
    assert len(store.get(str(fraud_payload["card1"]))) == 2


def test_fraud_explains_only_flagged_by_default(client, fraud_payload, trained):
    """Fast path: SHAP (the expensive step) only for review/block decisions."""
    from risklens.scorer import Scorer

    scorer = Scorer.load(trained["fraud"])
    body = client.post(
        "/v1/fraud/score", json={"transaction": fraud_payload, "record_history": False}
    ).json()
    assert (body["decision"] == "approve") == (body["reasons"] == [])
    # force a flag by lowering the threshold: now reasons must be present
    client.app.state.registry.get("fraud").scorer.threshold = 0.0
    try:
        flagged = client.post(
            "/v1/fraud/score", json={"transaction": fraud_payload, "record_history": False}
        ).json()
        assert flagged["decision"] == "block" and len(flagged["reasons"]) == 3
    finally:
        client.app.state.registry.get("fraud").scorer.threshold = scorer.threshold


def test_fraud_latency_budget(client, fraud_payload):
    """p95 < 50 ms incl. feature computation and SHAP reasons (in-process)."""
    for _ in range(5):  # warm-up
        client.post("/v1/fraud/score", json={"transaction": fraud_payload, "record_history": False})
    times = []
    for _ in range(40):
        t0 = time.perf_counter()
        r = client.post(
            "/v1/fraud/score", json={"transaction": fraud_payload, "record_history": False}
        )
        times.append((time.perf_counter() - t0) * 1000)
        assert r.status_code == 200
    assert np.percentile(times, 95) < 50, f"p95={np.percentile(times, 95):.1f} ms"


def test_model_not_loaded_returns_503():
    settings = Settings(
        model_source="local",
        credit_model_path="nope",
        fraud_model_path="nope",
        reload_interval_seconds=0,
        log_json=False,
    )
    with TestClient(create_app(settings)) as c:
        r = c.post(
            "/v1/credit/score",
            json={"features": {"AMT_INCOME_TOTAL": 1, "AMT_CREDIT": 1, "DAYS_BIRTH": -1}},
        )
        assert r.status_code == 503
        assert r.json()["error"] == "model_not_loaded"
        assert c.get("/health/ready").status_code == 503


def test_model_info_and_reload(client):
    info = client.get("/v1/model/info").json()
    assert info["credit"]["loaded"] and info["fraud"]["loaded"]
    assert "roc_auc" in info["credit"]["metrics"]
    assert client.post("/v1/model/reload").status_code == 401
    r = client.post("/v1/model/reload", headers={"X-Internal-Token": "secret"})
    assert r.status_code == 200 and set(r.json()) == {"credit", "fraud"}


def test_reference_and_drift(client, credit_payload):
    ref = client.get("/v1/model/reference", params={"task": "credit", "n": 50}).json()
    assert len(ref["records"]) == 50 and "score" in ref["records"][0]
    r = client.post(
        "/v1/monitoring/drift",
        json={"task": "credit", "records": [credit_payload] * 30, "scores": [0.5] * 30},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["drift_detected"] is True  # 30 identical rows are very different from the reference
    assert body["prediction_psi"] > 0.2


def test_metrics_endpoint(client, credit_payload):
    client.post("/v1/credit/score", json={"features": credit_payload})
    text = client.get("/metrics").text
    assert "risklens_prediction_probability_bucket" in text
    assert "risklens_model_loaded" in text
    assert 'route="/v1/credit/score"' in text


def test_shadow_mode(settings, trained, credit_payload):
    settings = settings.model_copy(update={"shadow_enabled": True})
    registry = ModelRegistry(settings)
    app = create_app(settings, registry=registry)
    with TestClient(app) as c:
        registry._challengers["credit"] = LoadedModel(
            "credit", Scorer.load(trained["credit"]), "99", "test"
        )
        body = c.post("/v1/credit/score", json={"features": credit_payload}).json()
        assert body["shadow"]["model_version"] == "99"
        assert abs(body["shadow"]["probability"] - body["probability"]) < 1e-9
