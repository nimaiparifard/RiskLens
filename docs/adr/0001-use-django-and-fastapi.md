# ADR 0001 – Django for the product API, FastAPI for model serving

* Status: accepted
* Date: 2026-10-03

## Context
RiskLens has two very different workloads:

1. **Product API** – users, roles, applications, decisions, an immutable audit log,
   admin screens, background jobs. Correctness, permissions and auditability matter more
   than raw latency.
2. **Model inference** – a CPU-bound, stateless call that must answer in < 50 ms (fraud) and
   needs the full ML stack (LightGBM, XGBoost, CatBoost, MLflow) in memory.

## Decision
* **Django + DRF** for (1): ORM + migrations, auth, admin, django-filter, drf-spectacular
  OpenAPI, Celery integration — all battle-tested.
* **FastAPI + Uvicorn** for (2): async I/O, pydantic request validation, tiny overhead,
  independent scaling (HPA on CPU) and independent deploys (hot reload by registry alias).
* They talk over HTTP through one client class (`apps/scoring/clients.py`) with timeouts,
  retries (5xx / connection errors only) and request-id propagation.

## Consequences
* + ML dependencies (≈1 GB) live only in the serving image; the backend stays light.
* + A model release never requires a backend deploy.
* − One extra network hop (~1–2 ms on the same host) and a second service to operate.
* − The backend must handle serving outages → `ServingUnavailable` → HTTP 503 with a clear
  error code; applications stay `pending` and the 15-minute Celery batch job scores them later.
