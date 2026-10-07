"""DVC stages ``validate`` and ``features``: raw tables -> validated -> dataset parquet."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from risklens.config import TrainConfig
from risklens.data import load, schemas, synthetic
from risklens.data.load import CREDIT_TEST_FILE

log = logging.getLogger(__name__)


def load_raw(cfg: TrainConfig, test_file: bool = False) -> dict[str, pd.DataFrame]:
    if cfg.task == "credit":
        if cfg.data.synthetic:
            tables = synthetic.make_credit_tables(cfg.data.synthetic_rows, cfg.seed)
            if test_file:
                tables["application"] = tables["application"].drop(columns=["TARGET"]).head(200)
            return tables
        app_file = CREDIT_TEST_FILE if test_file else load.CREDIT_FILES["application"]
        return load.load_credit_tables(
            cfg.data.raw_dir, cfg.data.nrows, cfg.feature_sources, app_file
        )
    if cfg.data.synthetic:
        return {
            "transactions": synthetic.make_fraud_transactions(cfg.data.synthetic_rows, cfg.seed)
        }
    return {"transactions": load.load_fraud_tables(cfg.data.raw_dir, cfg.data.nrows)}


def validate(
    cfg: TrainConfig, tables: dict[str, pd.DataFrame] | None = None, report: Path | None = None
) -> dict:
    """Fail loudly if raw data violates the schema."""
    tables = tables or load_raw(cfg)
    if cfg.task == "credit":
        schemas.validate_credit_tables(tables, with_target=cfg.target in tables["application"])
        app = tables["application"]
        summary = {
            "rows": len(app),
            "target_rate": float(app[cfg.target].mean()) if cfg.target in app else None,
            "days_employed_anomaly": int(
                (app["DAYS_EMPLOYED"] == schemas.DAYS_EMPLOYED_ANOMALY).sum()
            ),
            "tables": {k: len(v) for k, v in tables.items()},
        }
    else:
        tx = tables["transactions"]
        schemas.validate_fraud(tx)
        summary = {"rows": len(tx), "target_rate": float(tx[cfg.target].mean())}
    log.info("validation passed: %s", summary)
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_dataset(cfg: TrainConfig, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if cfg.task == "credit":
        from risklens.features.registry import build_credit_features

        return build_credit_features(tables, cfg.feature_sources)
    return tables["transactions"]


def run(cfg: TrainConfig, validate_first: bool = True) -> Path:
    tables = load_raw(cfg)
    if validate_first:
        validate(cfg, tables, Path(cfg.reports_dir) / "validation.json")
    ds = build_dataset(cfg, tables)
    out = Path(cfg.data.processed_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(out, index=False)
    log.info("wrote %s (%d rows x %d cols)", out, *ds.shape)
    if cfg.task == "credit" and cfg.data.processed_test_path:
        if cfg.data.synthetic:
            test_app = tables["application"].drop(columns=[cfg.target]).head(200)
        else:
            test_app = load.read_csv(Path(cfg.data.raw_dir) / CREDIT_TEST_FILE)
        test_tables = {**tables, "application": test_app}  # history tables are shared
        build_dataset(cfg, test_tables).to_parquet(cfg.data.processed_test_path, index=False)
        log.info("wrote %s (unlabelled applications for the demo)", cfg.data.processed_test_path)
    return out
