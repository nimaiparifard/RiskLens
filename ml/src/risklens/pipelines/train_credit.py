"""Credit-risk training entry point (DVC stage ``train``)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from risklens.config import TrainConfig
from risklens.data.split import load_or_create_split, stratified_split
from risklens.pipelines import features_stage
from risklens.pipelines.trainer import Trainer, TrainResult

log = logging.getLogger(__name__)

HISTORY_PREFIXES = (
    "BURO_",
    "ACTIVE_",
    "CLOSED_",
    "PREV_",
    "APPROVED_",
    "REFUSED_",
    "INSTAL_",
    "POS_",
    "CC_",
)


def load_dataset(cfg: TrainConfig) -> pd.DataFrame:
    path = Path(cfg.data.processed_path)
    if not path.exists():
        log.info("%s missing -> running features stage", path)
        features_stage.run(cfg)
    return pd.read_parquet(path)


def split_dataset(cfg: TrainConfig, df: pd.DataFrame) -> pd.DataFrame:
    ids, y = df[cfg.id_column], df[cfg.target]
    return load_or_create_split(
        Path(cfg.data.splits_path),
        ids,
        lambda: stratified_split(ids, y, cfg.split.valid_size, cfg.split.test_size, cfg.seed),
    )


def input_columns(df: pd.DataFrame, cfg: TrainConfig) -> tuple[list[str], list[str]]:
    """Raw application columns vs. history aggregates (supplied separately at serving)."""
    history = [c for c in df.columns if c.startswith(HISTORY_PREFIXES)]
    raw = [c for c in df.columns if c not in history and c != cfg.target]
    return raw, history


def run(cfg: TrainConfig) -> TrainResult:
    df = load_dataset(cfg)
    if cfg.data.nrows and len(df) > cfg.data.nrows:
        df = df.head(cfg.data.nrows)
    split = split_dataset(cfg, df)
    split_arr = df[cfg.id_column].map(split.set_index("id")["split"]).to_numpy()
    raw_cols, history_cols = input_columns(df, cfg)
    return Trainer(cfg).run(df, split_arr, raw_cols, history_cols)
