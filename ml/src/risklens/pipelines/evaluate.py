"""DVC stage ``evaluate``: re-score the FROZEN test set with the saved Scorer.

Independent of the training process, so it proves the saved artifact (not an
in-memory object) reproduces the reported numbers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from risklens.config import TrainConfig
from risklens.data.split import TEST
from risklens.evaluation.report import evaluate_predictions
from risklens.scorer import Scorer

log = logging.getLogger(__name__)


def run(cfg: TrainConfig, scorer_path: Path | None = None) -> dict[str, float]:
    scorer = Scorer.load(scorer_path or Path(cfg.model_dir) / "scorer.joblib")
    df = pd.read_parquet(cfg.data.processed_path)
    split = pd.read_parquet(cfg.data.splits_path)
    test_ids = set(split.loc[split["split"] == TEST, "id"])
    if cfg.task == "fraud":
        # history features need the full timeline; score everything, keep test rows
        df = df.sort_values("TransactionDT").reset_index(drop=True)
        prepared = scorer.prepare(df)
        mask = df[cfg.id_column].isin(test_ids).to_numpy()
        p = scorer.predict_proba_prepared(prepared[mask])
        frame = prepared[mask]
        y = df.loc[mask, cfg.target].to_numpy()
    else:
        test = df[df[cfg.id_column].isin(test_ids)].reset_index(drop=True)
        p = scorer.predict_proba(test)
        frame = scorer.prepare(test)
        y = test[cfg.target].to_numpy()
    out_dir = Path(cfg.reports_dir) / "evaluate"
    report = evaluate_predictions(y, p, scorer.threshold, cfg.costs, frame, out_dir=out_dir)
    report.to_json(out_dir / "metrics.json")
    log.info("evaluate: AUC=%.4f KS=%.4f", report.metrics["roc_auc"], report.metrics["ks"])
    (Path(cfg.reports_dir) / "eval_metrics.json").write_text(
        json.dumps(report.metrics, indent=2), encoding="utf-8"
    )
    return report.metrics


def latency_benchmark(scorer: Scorer, raw: pd.DataFrame, n: int = 200) -> dict[str, float]:
    """Single-row latency incl. features + SHAP reasons (what serving does)."""
    import time

    import numpy as np

    times = []
    for i in range(min(n, len(raw))):
        row = raw.iloc[[i]]
        t0 = time.perf_counter()
        scorer.score(row, explain=True)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }
