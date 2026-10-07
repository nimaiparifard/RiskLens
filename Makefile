# RiskLens – one-word commands.  `make help` lists them.
SHELL := /bin/bash
COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
PY := uv run
CREDIT := ml/configs/credit.yaml
FRAUD := ml/configs/fraud.yaml
export MLFLOW_TRACKING_URI ?= http://localhost:5000

.DEFAULT_GOAL := help
.PHONY: help setup up down logs ps migrate seed simulate data validate features train train-fraud evaluate \
        promote approve test test-ml test-serving test-backend test-frontend lint format types docs \
        bootstrap-models demo-data clean

help: ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# ------------------------------------------------------------------ setup
setup: ## install python (uv) + npm deps + pre-commit hooks
	uv sync --all-packages --all-extras
	cd frontend && npm install
	uv run pre-commit install
	@test -f .env || cp .env.example .env

# ----------------------------------------------------------------- docker
up: ## build + start the full stack
	$(COMPOSE) up -d --build
down: ## stop the stack
	$(COMPOSE) down
logs: ## follow logs
	$(COMPOSE) logs -f --tail=100
ps:
	$(COMPOSE) ps
migrate: ## django migrations
	$(COMPOSE) exec backend python manage.py migrate
seed: ## demo users + applications (scored)
	$(COMPOSE) exec backend python manage.py seed_demo --applications 80 --score
simulate: ## stream demo card transactions through the fraud API
	$(COMPOSE) exec backend python manage.py simulate_transactions --rate 3 --count 600

# --------------------------------------------------------------------- ML
data: ## download Kaggle data + version it with DVC
	$(PY) risklens download credit
	$(PY) risklens download fraud
	dvc add data/raw/home-credit data/raw/ieee-fraud
validate: ## pandera schema checks on raw data
	$(PY) risklens validate -c $(CREDIT)
features: ## build modelling datasets
	$(PY) risklens features -c $(CREDIT)
	$(PY) risklens features -c $(FRAUD)
train: ## train credit models (MLflow logged + registered)
	$(PY) risklens train -c $(CREDIT)
train-fraud: ## train fraud model
	$(PY) risklens train -c $(FRAUD)
evaluate: ## re-score the frozen test sets with the saved scorers
	$(PY) risklens evaluate -c $(CREDIT)
	$(PY) risklens evaluate -c $(FRAUD)
promote: ## compare latest version with champion -> challenger
	$(PY) risklens promote -c $(CREDIT)
	$(PY) risklens promote -c $(FRAUD)
approve: ## human approval: challenger -> champion
	$(PY) risklens promote -c $(CREDIT) --approve
bootstrap-models: ## first deployment: promote straight to champion
	$(PY) risklens promote -c $(CREDIT) --bootstrap
	$(PY) risklens promote -c $(FRAUD) --bootstrap
demo-data: ## export real (unlabelled) demo rows for the backend seeders
	$(PY) python scripts/export_demo_data.py
demo: ## record the README demo from the running app -> docs/media/demo.{gif,mp4}
	cd frontend && E2E_BASE_URL=$${E2E_BASE_URL:-http://localhost:3000} node e2e/record_demo.mjs
	$(PY) python scripts/make_demo_media.py --fps 8 --width 880
screenshots: ## refresh docs/screenshots from the running app
	cd frontend && E2E_BASE_URL=$${E2E_BASE_URL:-http://localhost:3000} node e2e/screenshots.mjs
pipeline: ## run the whole DVC pipeline (validate -> features -> train -> evaluate)
	dvc repro

# ------------------------------------------------------------------ quality
test: test-ml test-serving test-backend test-frontend ## all tests
test-ml:
	cd ml && $(PY) pytest
test-serving:
	cd serving && $(PY) pytest
test-backend:
	cd backend && $(PY) pytest --cov --cov-fail-under=80
test-frontend:
	cd frontend && npm test
lint: ## ruff + mypy + eslint + tsc
	$(PY) ruff check .
	$(PY) ruff format --check .
	$(PY) mypy ml/src/risklens
	cd frontend && npm run lint && npm run typecheck
format:
	$(PY) ruff format . && $(PY) ruff check --fix .
types: ## regenerate frontend TS types from the Django OpenAPI schema
	cd backend && $(PY) python manage.py spectacular --file ../.cache/openapi.yaml
	cd frontend && npm run types
docs: ## build the OpenAPI schema into docs/
	cd backend && $(PY) python manage.py spectacular --file ../docs/openapi.yaml

clean:
	rm -rf .cache **/__pycache__ frontend/dist
