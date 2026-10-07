"""Fraud training entry point. TIME-BASED split (train past, test future)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from risklens.config import TrainConfig
from risklens.data.split import load_or_create_split, time_split
from risklens.pipelines import features_stage
from risklens.pipelines.trainer import Trainer, TrainResult

log = logging.getLogger(__name__)

# raw fields a payment gateway would send at authorisation time
RAW_PREFIXES = ("card", "addr", "dist", "C", "D", "M", "V", "id_")
RAW_FIXED = [
    "TransactionID",
    "TransactionDT",
    "TransactionAmt",
    "ProductCD",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
]


def load_dataset(cfg: TrainConfig) -> pd.DataFrame:
    path = Path(cfg.data.processed_path)
    if not path.exists():
        features_stage.run(cfg)
    return pd.read_parquet(path)


def run(cfg: TrainConfig) -> TrainResult:
    df = (
        load_dataset(cfg)
        .sort_values(cfg.split.time_column or "TransactionDT")
        .reset_index(drop=True)
    )
    if cfg.data.nrows and len(df) > cfg.data.nrows:
        df = df.head(cfg.data.nrows)
    time_col = cfg.split.time_column or "TransactionDT"
    split = load_or_create_split(
        Path(cfg.data.splits_path),
        df[cfg.id_column],
        lambda: time_split(
            df[cfg.id_column], df[time_col], cfg.split.valid_size, cfg.split.test_size
        ),
    )
    split_arr = df[cfg.id_column].map(split.set_index("id")["split"]).to_numpy()
    raw_cols = [
        c for c in df.columns if c in RAW_FIXED or (c.startswith(RAW_PREFIXES) and c != cfg.target)
    ]
    return Trainer(cfg).run(df, split_arr, raw_cols, [])
