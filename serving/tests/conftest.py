"""Train tiny credit + fraud Scorers on synthetic data once per session."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from risklens.config import load_config
from risklens.pipelines import train_credit, train_fraud

CONFIGS = Path(__file__).resolve().parents[2] / "ml" / "configs"


def _common(tmp: Path, rows: int) -> list[str]:
    return [
        f"data.processed_path={tmp}/ds.parquet",
        f"data.splits_path={tmp}/splits.parquet",
        f"reports_dir={tmp}/reports",
        f"model_dir={tmp}/model",
        "data.processed_test_path=null",
        "mlflow=false",
        "data.synthetic=true",
        f"data.synthetic_rows={rows}",
        "fairness.enabled=false",
        "robustness_check=false",
    ]


@pytest.fixture(scope="session")
def trained(tmp_path_factory) -> dict[str, Path]:
    credit_dir = tmp_path_factory.mktemp("credit")
    fraud_dir = tmp_path_factory.mktemp("fraud")
    fast_lgbm = "models=[{name: lightgbm, params: {n_estimators: 120, learning_rate: 0.1, early_stopping_rounds: 30}}]"
    credit_cfg = load_config(
        CONFIGS / "ci.yaml", [*_common(credit_dir, 1500), fast_lgbm, "final_model=lightgbm"]
    )
    fraud_cfg = load_config(
        CONFIGS / "fraud.yaml", [*_common(fraud_dir, 12000), fast_lgbm, "selection.top_k=40"]
    )
    c = train_credit.run(credit_cfg)
    f = train_fraud.run(fraud_cfg)
    return {
        "credit": c.scorer_path,
        "fraud": f.scorer_path,
        "credit_data": Path(credit_cfg.data.processed_path),
        "fraud_data": Path(fraud_cfg.data.processed_path),
    }


@pytest.fixture
def settings(trained) -> Settings:
    return Settings(
        model_source="local",
        credit_model_path=str(trained["credit"]),
        fraud_model_path=str(trained["fraud"]),
        internal_token="secret",
        reload_interval_seconds=0,
        log_json=False,
    )


@pytest.fixture
def client(settings) -> TestClient:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def credit_payload(trained) -> dict:
    df = pd.read_parquet(trained["credit_data"]).drop(columns=["TARGET"]).head(1)
    row = df.iloc[0].to_dict()
    return {
        k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()
    }


@pytest.fixture
def fraud_payload(trained) -> dict:
    df = pd.read_parquet(trained["fraud_data"]).drop(columns=["isFraud"]).tail(1)
    row = df.iloc[0].to_dict()
    return {
        k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v)) for k, v in row.items()
    }
