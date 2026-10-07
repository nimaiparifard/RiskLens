import pytest
from rest_framework.test import APIClient

from .factories import UserFactory

pytestmark = pytest.mark.django_db


def test_login_returns_tokens_and_role():
    UserFactory(username="alice", role="officer")
    r = APIClient().post(
        "/api/auth/token/", {"username": "alice", "password": "pass12345!"}, format="json"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["access"] and body["refresh"]
    assert body["user"]["role"] == "officer"


def test_bad_password():
    UserFactory(username="bob")
    r = APIClient().post("/api/auth/token/", {"username": "bob", "password": "nope"}, format="json")
    assert r.status_code == 401


def test_me_requires_auth(officer_client):
    assert APIClient().get("/api/auth/me/").status_code == 401
    assert officer_client.get("/api/auth/me/").json()["username"] == "officer"


def test_refresh_rotation():
    UserFactory(username="carol")
    c = APIClient()
    tokens = c.post(
        "/api/auth/token/", {"username": "carol", "password": "pass12345!"}, format="json"
    ).json()
    r = c.post("/api/auth/token/refresh/", {"refresh": tokens["refresh"]}, format="json")
    assert r.status_code == 200 and "refresh" in r.json()
    # rotated + blacklisted -> old refresh token no longer valid
    assert (
        c.post(
            "/api/auth/token/refresh/", {"refresh": tokens["refresh"]}, format="json"
        ).status_code
        == 401
    )


def test_health_and_docs(client):
    assert client.get("/api/health/").json()["status"] == "ok"
    assert client.get("/api/schema/").status_code == 200
