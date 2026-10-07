from __future__ import annotations

import pytest
from rest_framework.test import APIClient

from apps.core.exceptions import ServingUnavailable

from .factories import ApplicationFactory, UserFactory


class FakeServingClient:
    """Stands in for the FastAPI model server."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []
        self.credit_result = {
            "task": "credit",
            "probability": 0.21,
            "score_points": 553,
            "grade": "D",
            "decision": "reject",
            "threshold": 0.18,
            "model_name": "risklens-credit",
            "model_version": "3",
            "latency_ms": 7.5,
            "reasons": [
                {
                    "feature": "EXT_SOURCE_2",
                    "value": 0.1,
                    "contribution": 0.8,
                    "direction": "increases_risk",
                    "text": "Low external credit bureau score (bureau 2)",
                }
            ],
            "shadow": None,
        }
        self.fraud_result = {
            "task": "fraud",
            "probability": 0.93,
            "grade": "HIGH",
            "decision": "block",
            "threshold": 0.4,
            "model_version": "5",
            "latency_ms": 4.2,
            "reasons": [],
        }
        self.drift_result = {
            "model_version": "3",
            "feature_psi": {"EXT_SOURCE_2": 0.35, "AMT_CREDIT": 0.01},
            "prediction_psi": 0.05,
            "max_psi": 0.35,
            "drifted_features": ["EXT_SOURCE_2"],
            "drift_detected": True,
        }
        self.unavailable = False

    def _check(self):
        if self.unavailable:
            raise ServingUnavailable("model server unavailable")

    def score_credit(self, features, explain=True):
        self._check()
        self.calls.append(("credit", features))
        return dict(self.credit_result)

    def score_credit_batch(self, items):
        self._check()
        return [dict(self.credit_result) for _ in items]

    def score_fraud(self, transaction, explain=True):
        self._check()
        self.calls.append(("fraud", transaction))
        return dict(self.fraud_result)

    def drift(self, task, records, scores, psi_threshold):
        self._check()
        self.calls.append(("drift", len(records)))
        return dict(self.drift_result)

    def reference(self, task, n=2000):
        return {"records": []}

    def model_info(self):
        return {"credit": {"loaded": True, "version": "3"}}

    def reload(self, task=None):
        return {"credit": "3", "fraud": "5"}


@pytest.fixture
def serving(monkeypatch) -> FakeServingClient:
    fake = FakeServingClient()
    for module in (
        "apps.scoring.services",
        "apps.transactions.services",
        "apps.monitoring.services",
        "apps.mlmodels.api.views",
    ):
        monkeypatch.setattr(f"{module}.get_serving_client", lambda: fake)
    return fake


def _client(user) -> APIClient:
    c = APIClient()
    c.force_authenticate(user)
    return c


@pytest.fixture
def officer(db):
    return UserFactory(username="officer", role="officer")


@pytest.fixture
def admin_user(db):
    return UserFactory(username="admin", role="admin")


@pytest.fixture
def viewer(db):
    return UserFactory(username="viewer", role="viewer")


@pytest.fixture
def officer_client(officer):
    return _client(officer)


@pytest.fixture
def admin_client(admin_user):
    return _client(admin_user)


@pytest.fixture
def viewer_client(viewer):
    return _client(viewer)


@pytest.fixture
def application(db):
    return ApplicationFactory()
