"""Export realistic demo rows for the backend seeders.

* applications.json  <- data/processed/credit_features_test.parquet
  (Kaggle application_test.csv: real applications WITHOUT labels, plus their
  bureau / previous-loan aggregates -> the 'history' the model expects)
* transactions.json  <- last (test-period) rows of data/processed/fraud_features.parquet

    uv run python scripts/export_demo_data.py --applications 300 --transactions 2000
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from risklens.pipelines.train_credit import HISTORY_PREFIXES
from risklens.pipelines.train_fraud import RAW_FIXED, RAW_PREFIXES

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "backend" / "demo_data"


def clean(record: dict) -> dict:
    out = {}
    for k, v in record.items():
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        out[k] = v.item() if hasattr(v, "item") else v
    return out


def export_applications(n: int, seed: int) -> int:
    path = ROOT / "data" / "processed" / "credit_features_test.parquet"
    df = pd.read_parquet(path).sample(n, random_state=seed)
    history_cols = [c for c in df.columns if c.startswith(HISTORY_PREFIXES)]
    app_cols = [c for c in df.columns if c not in history_cols]
    rows = []
    for _, r in df.iterrows():
        rec = r.to_dict()
        rows.append(
            {
                "SK_ID_CURR": int(rec["SK_ID_CURR"]),
                "features": clean({k: rec[k] for k in app_cols if k != "SK_ID_CURR"}),
                "history": clean({k: rec[k] for k in history_cols}),
            }
        )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "applications.json").write_text(json.dumps(rows), encoding="utf-8")
    return len(rows)


def export_transactions(n: int) -> int:
    path = ROOT / "data" / "processed" / "fraud_features.parquet"
    df = pd.read_parquet(path).sort_values("TransactionDT").tail(n)
    cols = [c for c in df.columns if c in RAW_FIXED or c.startswith(RAW_PREFIXES)]
    rows = [clean(r) for r in df[cols].to_dict(orient="records")]
    (OUT / "transactions.json").write_text(json.dumps(rows), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--applications", type=int, default=300)
    ap.add_argument("--transactions", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print("applications:", export_applications(a.applications, a.seed))
    print("transactions:", export_transactions(a.transactions))
