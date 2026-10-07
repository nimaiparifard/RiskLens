import httpx
import pytest

from apps.audit.models import AuditLog
from apps.core.exceptions import ServingRejected, ServingUnavailable
from apps.mlmodels.clients import MlflowRegistryClient
from apps.monitoring.models import DriftReport
from apps.monitoring.services import run_drift_report
from apps.scoring.clients import ServingClient
from apps.transactions.models import Transaction

from .factories import ApplicationFactory

pytestmark = pytest.mark.django_db

TX = {
    "TransactionID": 1,
    "TransactionDT": 86400,
    "TransactionAmt": 120.5,
    "card1": 1234,
    "ProductCD": "W",
    "P_emaildomain": "gmail.com",
    "V258": 1.0,
}


def test_score_transaction_blocks(officer_client, serving):
    r = officer_client.post("/api/transactions/score/", TX, format="json")
    assert r.status_code == 201, r.json()
    body = r.json()
    assert body["status"] == "blocked" and body["is_flagged"] is True
    assert serving.calls[0][1]["V258"] == 1.0  # raw extra fields forwarded
    assert officer_client.get("/api/transactions/?flagged=true").json()["count"] == 1


def test_transaction_feed_polling(officer_client, serving):
    officer_client.post("/api/transactions/score/", TX, format="json")
    first = Transaction.objects.get()
    officer_client.post("/api/transactions/score/", {**TX, "TransactionID": 2}, format="json")
    r = officer_client.get(f"/api/transactions/?since_id={first.id}")
    assert r.json()["count"] == 1


def test_stats_latency_is_a_recent_window(officer_client, serving):
    from datetime import timedelta

    from django.utils import timezone

    serving.fraud_result["latency_ms"] = 300.0  # old, slow
    officer_client.post("/api/transactions/score/", TX, format="json")
    Transaction.objects.update(created_at=timezone.now() - timedelta(hours=3))
    serving.fraud_result["latency_ms"] = 10.0  # recent, fast
    for i in (2, 3):
        officer_client.post("/api/transactions/score/", {**TX, "TransactionID": i}, format="json")
    stats = officer_client.get("/api/transactions/stats/").json()
    assert stats["total"] == 3 and stats["avg_latency_ms"] == 10.0
    assert stats["latency_window_minutes"] == 60


def test_review_transaction(officer_client, serving):
    tx_id = officer_client.post("/api/transactions/score/", TX, format="json").json()["id"]
    r = officer_client.post(f"/api/transactions/{tx_id}/review/", {"label": "fraud"}, format="json")
    assert r.json()["status"] == "confirmed_fraud"
    assert AuditLog.objects.filter(action="transaction.reviewed").exists()


def test_drift_report_skipped_without_data(serving):
    assert run_drift_report("credit").status == DriftReport.Status.SKIPPED


def test_drift_report_detects_drift(settings, officer_client, serving):
    settings.DRIFT_MIN_SAMPLES = 3
    for app in ApplicationFactory.create_batch(4):
        officer_client.post(f"/api/applications/{app.id}/score/")
    report = run_drift_report("credit")
    assert report.status == DriftReport.Status.DRIFT
    assert report.drifted_features == ["EXT_SOURCE_2"] and report.n_samples == 4
    assert ("drift", 4) in serving.calls
    assert AuditLog.objects.filter(action="drift.detected").exists()
    r = officer_client.get("/api/monitoring/drift/?task=credit").json()
    assert r["results"][0]["max_psi"] == 0.35


def test_run_drift_requires_admin(officer_client, admin_client, serving):
    assert (
        officer_client.post(
            "/api/monitoring/drift/run/", {"task": "credit"}, format="json"
        ).status_code
        == 403
    )
    assert (
        admin_client.post(
            "/api/monitoring/drift/run/", {"task": "credit"}, format="json"
        ).status_code
        == 201
    )


def test_models_endpoint_with_mlflow_down(officer_client, serving, monkeypatch):
    monkeypatch.setattr(
        MlflowRegistryClient,
        "_get",
        lambda *a, **k: (_ for _ in ()).throw(
            __import__("apps.core.exceptions", fromlist=["x"]).ExternalServiceError("down")
        ),
    )
    body = officer_client.get("/api/models/").json()
    assert body["registry"]["credit"]["error"] == "down"
    assert body["serving"]["credit"]["version"] == "3"


def test_models_reload_admin_only(officer_client, admin_client, serving):
    assert officer_client.post("/api/models/reload/").status_code == 403
    assert admin_client.post("/api/models/reload/").json() == {"credit": "3", "fraud": "5"}


def test_mlflow_registry_client_parses_rest_api():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("registered-models/get"):
            return httpx.Response(
                200, json={"registered_model": {"aliases": [{"alias": "champion", "version": "2"}]}}
            )
        if path.endswith("model-versions/search"):
            return httpx.Response(
                200,
                json={
                    "model_versions": [
                        {"version": "1", "run_id": "r1"},
                        {"version": "2", "run_id": "r2"},
                    ]
                },
            )
        if path.endswith("runs/get"):
            return httpx.Response(
                200,
                json={
                    "run": {
                        "data": {
                            "metrics": [
                                {"key": "roc_auc", "value": 0.79},
                                {"key": "other", "value": 1},
                            ]
                        }
                    }
                },
            )
        return httpx.Response(404)

    summary = MlflowRegistryClient(
        "http://mlflow", transport=httpx.MockTransport(handler)
    ).model_summary("m")
    assert summary["aliases"] == {"champion": "2"}
    assert summary["versions"][0]["version"] == "2" and summary["versions"][0]["is_champion"]
    assert summary["versions"][0]["metrics"] == {"roc_auc": 0.79}


# ------------------------------------------------------------ ServingClient
def test_serving_client_retries_5xx_then_succeeds():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(502)
        return httpx.Response(200, json={"probability": 0.1})

    client = ServingClient("http://s", retries=3, transport=httpx.MockTransport(handler))
    assert client.score_credit({"a": 1})["probability"] == 0.1
    assert attempts["n"] == 2


def test_serving_client_does_not_retry_4xx():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(422, json={"detail": "bad"})

    with pytest.raises(ServingRejected):
        ServingClient("http://s", retries=3, transport=httpx.MockTransport(handler)).score_credit(
            {}
        )
    assert attempts["n"] == 1


def test_serving_client_unreachable():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ServingUnavailable):
        ServingClient("http://s", retries=2, transport=httpx.MockTransport(handler)).score_credit(
            {}
        )


def test_request_id_propagated(officer_client, application, serving):
    r = officer_client.post(
        f"/api/applications/{application.id}/score/", HTTP_X_REQUEST_ID="abc123"
    )
    assert r["X-Request-ID"] == "abc123"
    assert AuditLog.objects.get(action="application.scored").request_id == "abc123"
