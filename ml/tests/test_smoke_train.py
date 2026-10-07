"""End-to-end smoke train on synthetic data (CI: < 1 minute, CPU only)."""

from __future__ import annotations

import pandas as pd
import pytest

from risklens.config import load_config
from risklens.pipelines import evaluate, train_credit, train_fraud
from risklens.scorer import Scorer

from .conftest import CONFIGS


def _cfg(tmp_path, task: str, rows: int):
    overrides = [
        f"data.processed_path={tmp_path}/ds.parquet",
        f"data.splits_path={tmp_path}/splits.parquet",
        f"reports_dir={tmp_path}/reports",
        f"model_dir={tmp_path}/model",
        "data.processed_test_path=null",
        "mlflow=false",
        "data.synthetic=true",
        f"data.synthetic_rows={rows}",
    ]
    if task == "credit":
        return load_config(CONFIGS / "ci.yaml", overrides)
    return load_config(
        CONFIGS / "fraud.yaml",
        [
            *overrides,
            "models=[{name: lightgbm, params: {n_estimators: 150, learning_rate: 0.1, early_stopping_rounds: 30}}]",
            "selection.top_k=40",
            "cv_folds=2",
        ],
    )


@pytest.mark.slow
def test_smoke_train_credit(tmp_path):
    cfg = _cfg(tmp_path, "credit", 1200)
    result = train_credit.run(cfg)
    assert result.test_metrics["roc_auc"] > 0.6
    assert set(result.model_table["model"]) == {"logreg_woe", "lightgbm", "xgboost", "stacking"}
    assert (tmp_path / "reports" / "fairness.md").exists()
    assert (tmp_path / "reports" / "plots" / "test_roc.png").exists()

    scorer = Scorer.load(result.scorer_path)
    df = pd.read_parquet(cfg.data.processed_path).head(3)
    res = scorer.score(df[scorer.input_columns + scorer.history_columns])
    assert len(res) == 3 and len(res[0].reasons) == cfg.explain.top_k
    assert res[0].decision in {"approve", "review", "reject"}
    assert scorer.reference is not None and "score" in scorer.reference

    metrics = evaluate.run(cfg)
    assert metrics["roc_auc"] == pytest.approx(result.test_metrics["roc_auc"], abs=1e-6)


@pytest.mark.slow
def test_smoke_train_fraud(tmp_path):
    cfg = _cfg(tmp_path, "fraud", 12000)  # enough frauds for a stable time-split AUC
    result = train_fraud.run(cfg)
    assert result.test_metrics["roc_auc"] > 0.6
    scorer = Scorer.load(result.scorer_path)
    tx = pd.read_parquet(cfg.data.processed_path).tail(2)
    res = scorer.score(tx[scorer.input_columns])
    assert res[0].decision in {"approve", "review", "block"}
