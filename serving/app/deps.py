"""FastAPI dependencies (state lives on app.state, created in the lifespan)."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from app.config import Settings
from app.feature_service import CreditFeatureService, FraudFeatureService
from app.model_loader import ModelRegistry


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def get_credit_service(request: Request) -> CreditFeatureService:
    return request.app.state.credit_service


def get_fraud_service(request: Request) -> FraudFeatureService:
    return request.app.state.fraud_service


def require_internal_token(request: Request, x_internal_token: str | None = Header(None)) -> None:
    if x_internal_token != request.app.state.settings.internal_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal token")
