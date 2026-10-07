"""Load raw Kaggle tables into pandas (fast pyarrow CSV engine)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CREDIT_FILES: dict[str, str] = {
    "application": "application_train.csv",
    "bureau": "bureau.csv",
    "bureau_balance": "bureau_balance.csv",
    "previous": "previous_application.csv",
    "installments": "installments_payments.csv",
    "pos_cash": "POS_CASH_balance.csv",
    "credit_card": "credit_card_balance.csv",
}
CREDIT_TEST_FILE = "application_test.csv"

FRAUD_FILES: dict[str, str] = {
    "transaction": "train_transaction.csv",
    "identity": "train_identity.csv",
}


def read_csv(path: Path, nrows: int | None = None) -> pd.DataFrame:
    if nrows is not None:
        df = pd.read_csv(path, nrows=nrows)
    else:
        try:
            df = pd.read_csv(path, engine="pyarrow")
        except Exception:  # pragma: no cover - pyarrow edge cases
            df = pd.read_csv(path, low_memory=False)
    return downcast(df)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    """float64 -> float32 halves memory on the 27M-row balance tables."""
    floats = df.select_dtypes(include=["float64"]).columns
    if len(floats):
        df[floats] = df[floats].astype(np.float32)
    return df


def load_credit_tables(
    raw_dir: Path,
    nrows: int | None = None,
    sources: list[str] | None = None,
    application_file: str = CREDIT_FILES["application"],
) -> dict[str, pd.DataFrame]:
    """Load the Home Credit tables. With ``nrows`` only the first N applications
    (and their related history rows) are kept."""
    raw_dir = Path(raw_dir)
    needed = {"application"} | set(sources or CREDIT_FILES)
    if "bureau" in needed:
        needed.add("bureau_balance")
    tables: dict[str, pd.DataFrame] = {}
    app = read_csv(raw_dir / application_file, nrows=nrows)
    tables["application"] = app
    ids = set(app["SK_ID_CURR"]) if nrows is not None else None
    for name, fname in CREDIT_FILES.items():
        if name == "application" or name not in needed:
            continue
        path = raw_dir / fname
        if not path.exists():
            log.warning("missing table %s – features from it will be skipped", path)
            continue
        log.info("loading %s", fname)
        df = read_csv(path)
        if ids is not None and "SK_ID_CURR" in df.columns:
            df = df[df["SK_ID_CURR"].isin(ids)]
        tables[name] = df
    if ids is not None and "bureau_balance" in tables and "bureau" in tables:
        bb = tables["bureau_balance"]
        tables["bureau_balance"] = bb[
            bb["SK_ID_BUREAU"].isin(set(tables["bureau"]["SK_ID_BUREAU"]))
        ]
    return tables


def load_fraud_tables(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """train_transaction LEFT JOIN train_identity on TransactionID."""
    raw_dir = Path(raw_dir)
    tx = read_csv(raw_dir / FRAUD_FILES["transaction"], nrows=nrows)
    id_path = raw_dir / FRAUD_FILES["identity"]
    if id_path.exists():
        ident = read_csv(id_path)
        ident.columns = [c.replace("-", "_") for c in ident.columns]
        tx = tx.merge(ident, on="TransactionID", how="left")
    return tx.sort_values("TransactionDT").reset_index(drop=True)
