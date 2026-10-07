"""End-to-end smoke test against a running stack (docker compose or scripts/run_local.ps1).

    python scripts/smoke_e2e.py --api http://localhost:8000 --password risklens-demo

Exercises: login (JWT) -> list -> create -> score (Django -> serving -> SHAP) ->
decision with override -> audit trail -> fraud check -> models registry -> drift run.
Exits non-zero on the first failed expectation.
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--password", default="risklens-demo")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.api, timeout=60)

    def login(user: str) -> dict[str, str]:
        r = c.post("/api/auth/token/", json={"username": user, "password": a.password})
        r.raise_for_status()
        return {"Authorization": f"Bearer {r.json()['access']}"}

    def check(cond: bool, msg: str) -> None:
        print(("PASS " if cond else "FAIL ") + msg)
        if not cond:
            sys.exit(1)

    officer, admin, viewer = login("officer"), login("admin"), login("viewer")

    apps = c.get("/api/applications/?page_size=5", headers=officer).json()
    check(apps["count"] > 0, f"applications listed: {apps['count']}")

    new = c.post(
        "/api/applications/",
        headers=officer,
        json={
            "applicant": {"full_name": "E2E Tester", "email": "e2e@example.com"},
            "features": {
                "AMT_INCOME_TOTAL": 90000,
                "AMT_CREDIT": 900000,
                "AMT_ANNUITY": 45000,
                "DAYS_BIRTH": -8500,
                "DAYS_EMPLOYED": -200,
                "EXT_SOURCE_2": 0.08,
                "EXT_SOURCE_3": 0.15,
                "NAME_CONTRACT_TYPE": "Cash loans",
                "NAME_EDUCATION_TYPE": "Secondary / secondary special",
            },
        },
    )
    check(new.status_code == 201, f"application created {new.json().get('reference')}")
    app_id = new.json()["id"]
    check(
        c.post(f"/api/applications/{app_id}/score/", headers=viewer).status_code == 403,
        "viewer cannot score",
    )

    t0 = time.perf_counter()
    pred = c.post(f"/api/applications/{app_id}/score/", headers=officer).json()
    rt = (time.perf_counter() - t0) * 1000
    check(
        "probability" in pred,
        f"scored: PD={pred['probability']:.3f} grade={pred['grade']} "
        f"rec={pred['decision']} model v{pred['model_version']} serving={pred['latency_ms']:.1f}ms round-trip={rt:.0f}ms",
    )
    check(
        len(pred["shap_reasons"]) == 5,
        "5 SHAP reasons: " + " | ".join(r["text"] for r in pred["shap_reasons"]),
    )

    opposite = "approve" if pred["decision"] != "approve" else "reject"
    r = c.post(
        f"/api/applications/{app_id}/decision/", headers=officer, json={"decision": opposite}
    )
    check(r.status_code == 400, "override without comment rejected")
    r = c.post(
        f"/api/applications/{app_id}/decision/",
        headers=officer,
        json={"decision": opposite, "comment": "E2E: documents verified"},
    )
    check(
        r.status_code == 200 and r.json()["is_override"],
        f"override recorded -> {r.json()['status']}",
    )
    trail = [
        e["action"] for e in c.get(f"/api/applications/{app_id}/audit/", headers=officer).json()
    ]
    check(
        trail[:3] == ["decision.made", "application.scored", "application.created"],
        f"audit trail {trail}",
    )

    tx = c.post(
        "/api/transactions/score/",
        headers=officer,
        json={
            "TransactionID": int(time.time()),
            "TransactionDT": 16_000_000,
            "TransactionAmt": 3500.0,
            "ProductCD": "C",
            "card1": 4321,
            "addr1": 299.0,
            "D1": 0.0,
            "P_emaildomain": "protonmail.com",
            "DeviceType": "mobile",
        },
    )
    check(
        tx.status_code == 201,
        f"fraud check: p={tx.json()['probability']:.4f} -> {tx.json()['status']} "
        f"({tx.json()['latency_ms']:.1f} ms in serving)",
    )

    models = c.get("/api/models/", headers=officer).json()
    credit = models["registry"]["credit"]
    check(
        "versions" in credit and credit["versions"],
        f"registry versions: credit {[v['version'] for v in credit['versions']]} "
        f"aliases {credit['aliases']}, fraud aliases {models['registry']['fraud']['aliases']}",
    )

    drift = c.post(
        "/api/monitoring/drift/run/", headers=admin, json={"task": "credit", "hours": 24}
    )
    d = drift.json()
    check(
        drift.status_code == 201,
        f"drift report: status={d['status']} n={d['n_samples']} max_psi={d['max_psi']}",
    )
    print("\nall end-to-end checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
