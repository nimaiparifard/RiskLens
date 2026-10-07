import pytest

from apps.applicants.models import Application
from apps.audit.models import AuditLog
from apps.scoring.models import Prediction

from .factories import ApplicationFactory

pytestmark = pytest.mark.django_db

NEW = {
    "applicant": {"full_name": "Jane Roe", "email": "jane@example.com"},
    "features": {
        "AMT_INCOME_TOTAL": 120000,
        "AMT_CREDIT": 300000,
        "DAYS_BIRTH": -11000,
        "EXT_SOURCE_2": 0.5,
    },
}


def test_create_application(officer_client):
    r = officer_client.post("/api/applications/", NEW, format="json")
    assert r.status_code == 201, r.json()
    body = r.json()
    assert body["reference"].startswith("APP-") and body["status"] == "pending"
    assert AuditLog.objects.filter(action="application.created").count() == 1


def test_create_validates_features(officer_client):
    bad = {**NEW, "features": {"AMT_INCOME_TOTAL": 0, "AMT_CREDIT": 1, "DAYS_BIRTH": -1}}
    assert officer_client.post("/api/applications/", bad, format="json").status_code == 400
    missing = {**NEW, "features": {"AMT_CREDIT": 1}}
    r = officer_client.post("/api/applications/", missing, format="json")
    assert r.status_code == 400 and "DAYS_BIRTH" in str(r.json())


def test_viewer_cannot_create_but_can_list(viewer_client, application):
    assert viewer_client.post("/api/applications/", NEW, format="json").status_code == 403
    r = viewer_client.get("/api/applications/")
    assert r.status_code == 200 and r.json()["count"] == 1


def test_list_filters(officer_client):
    ApplicationFactory(status="pending")
    ApplicationFactory(status="approved", latest_grade="A")
    ApplicationFactory(status="rejected", latest_grade="E")
    assert officer_client.get("/api/applications/?status=pending").json()["count"] == 1
    assert (
        officer_client.get("/api/applications/?status=approved&status=rejected").json()["count"]
        == 2
    )
    assert officer_client.get("/api/applications/?grade=E").json()["count"] == 1


def test_score_stores_prediction_with_reasons(officer_client, application, serving):
    r = officer_client.post(f"/api/applications/{application.id}/score/")
    assert r.status_code == 201, r.json()
    pred = Prediction.objects.get()
    assert pred.model_version == "3" and pred.grade == "D"
    assert pred.shap_reasons[0]["feature"] == "EXT_SOURCE_2"
    assert pred.input_payload["BURO_COUNT"] == 2  # history aggregates forwarded to the model
    application.refresh_from_db()
    assert application.status == "scored" and application.recommendation == "reject"
    assert serving.calls[0][1]["AMT_CREDIT"] == 400000
    assert AuditLog.objects.filter(
        action="application.scored", target_id=str(application.id)
    ).exists()
    detail = officer_client.get(f"/api/applications/{application.id}/").json()
    assert detail["latest_prediction"]["probability"] == 0.21


def test_score_when_serving_down_returns_503(officer_client, application, serving):
    serving.unavailable = True
    r = officer_client.post(f"/api/applications/{application.id}/score/")
    assert r.status_code == 503
    assert r.json()["error"] == "serving_unavailable"
    assert not Prediction.objects.exists()


def test_viewer_cannot_score(viewer_client, application, serving):
    assert viewer_client.post(f"/api/applications/{application.id}/score/").status_code == 403


def _scored(officer_client, application):
    officer_client.post(f"/api/applications/{application.id}/score/")


def test_decision_requires_score(officer_client, application):
    r = officer_client.post(
        f"/api/applications/{application.id}/decision/", {"decision": "approve"}, format="json"
    )
    assert r.status_code == 409


def test_decision_following_recommendation(officer_client, application, serving):
    _scored(officer_client, application)
    r = officer_client.post(
        f"/api/applications/{application.id}/decision/", {"decision": "reject"}, format="json"
    )
    assert r.status_code == 200
    assert r.json()["status"] == "rejected" and r.json()["is_override"] is False


def test_override_needs_comment_and_is_audited(officer_client, application, serving):
    _scored(officer_client, application)  # model says reject
    url = f"/api/applications/{application.id}/decision/"
    r = officer_client.post(url, {"decision": "approve"}, format="json")
    assert r.status_code == 400 and r.json()["recommendation"] == "reject"
    r = officer_client.post(
        url, {"decision": "approve", "comment": "verified new employment contract"}, format="json"
    )
    assert r.status_code == 200 and r.json()["is_override"] is True
    log = AuditLog.objects.get(action="decision.made")
    assert log.payload["is_override"] is True and log.actor.username == "officer"
    trail = officer_client.get(f"/api/applications/{application.id}/audit/").json()
    assert [e["action"] for e in trail][:2] == ["decision.made", "application.scored"]


def test_final_decision_change_requires_admin(officer_client, admin_client, application, serving):
    _scored(officer_client, application)
    url = f"/api/applications/{application.id}/decision/"
    officer_client.post(url, {"decision": "reject"}, format="json")
    assert (
        officer_client.post(url, {"decision": "approve", "comment": "x"}, format="json").status_code
        == 403
    )
    r = admin_client.post(url, {"decision": "approve", "comment": "appeal accepted"}, format="json")
    assert r.status_code == 200 and r.json()["status"] == "approved"
    assert AuditLog.objects.filter(action="decision.overridden").count() == 1


def test_cannot_rescore_decided(officer_client, application, serving):
    _scored(officer_client, application)
    officer_client.post(
        f"/api/applications/{application.id}/decision/", {"decision": "reject"}, format="json"
    )
    assert officer_client.post(f"/api/applications/{application.id}/score/").status_code == 409


def test_stats(officer_client, serving):
    a = ApplicationFactory()
    ApplicationFactory()
    _scored(officer_client, a)
    stats = officer_client.get("/api/applications/stats/").json()
    assert (
        stats["total"] == 2 and stats["by_status"]["pending"] == 1 and stats["by_grade"]["D"] == 1
    )


def test_batch_score_task(serving):
    from apps.scoring.tasks import batch_score_pending

    ApplicationFactory.create_batch(3)
    assert batch_score_pending.delay().get() == 3
    assert Application.objects.filter(status="scored").count() == 3
    # batch scores are audited and carry reasons like interactive ones (compliance)
    assert AuditLog.objects.filter(action="application.scored", payload__batch=True).count() == 3
    assert all(p.shap_reasons for p in Prediction.objects.all())
