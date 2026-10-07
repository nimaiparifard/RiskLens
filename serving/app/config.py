"""12-factor settings for the model server (env vars, see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    # where models come from: MLflow registry (by alias) or local joblib files
    model_source: Literal["mlflow", "local"] = "mlflow"
    mlflow_tracking_uri: str | None = None
    credit_model_name: str = "risklens-credit"
    fraud_model_name: str = "risklens-fraud"
    model_alias: str = "champion"
    challenger_alias: str = "challenger"
    credit_model_path: str = "models/credit/scorer.joblib"
    fraud_model_path: str = "models/fraud/scorer.joblib"
    local_fallback: bool = True

    reload_interval_seconds: int = 300  # poll the registry for a new champion
    internal_token: str = "change-me"  # protects POST /v1/model/reload
    shadow_enabled: bool = False  # also score with the challenger and log both

    history_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/2"
    history_max_events: int = 500

    log_level: str = "INFO"
    log_json: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
