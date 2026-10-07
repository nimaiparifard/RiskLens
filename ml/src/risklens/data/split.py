"""Train / valid / test splits. The resulting id->split table is written to
parquet and versioned with DVC so the test set is FROZEN across experiments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TRAIN, VALID, TEST = "train", "valid", "test"


def stratified_split(
    ids: pd.Series, y: pd.Series, valid_size: float, test_size: float, seed: int
) -> pd.DataFrame:
    """Stratified 70/15/15 (by default) split."""
    idx = np.arange(len(ids))
    rest, test = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    rel_valid = valid_size / (1.0 - test_size)
    train, valid = train_test_split(
        rest, test_size=rel_valid, stratify=y.iloc[rest], random_state=seed
    )
    split = np.empty(len(ids), dtype=object)
    split[train], split[valid], split[test] = TRAIN, VALID, TEST
    return pd.DataFrame({"id": ids.to_numpy(), "split": split})


def time_split(
    ids: pd.Series, time: pd.Series, valid_size: float, test_size: float
) -> pd.DataFrame:
    """Train on the past, validate on the recent past, test on the future."""
    t = time.to_numpy()
    test_cut = np.quantile(t, 1.0 - test_size)
    valid_cut = np.quantile(t, 1.0 - test_size - valid_size)
    split = np.where(t >= test_cut, TEST, np.where(t >= valid_cut, VALID, TRAIN))
    return pd.DataFrame({"id": ids.to_numpy(), "split": split})


def ordered_split(ids: pd.Series, holdout: float = 0.15) -> pd.Series:
    """Out-of-time style robustness split: lowest ids train, highest ids hold out.

    Home Credit has no application date, but SK_ID_CURR is assigned
    sequentially, so id order is a reasonable time proxy.
    """
    cut = ids.quantile(1.0 - holdout)
    return pd.Series(np.where(ids >= cut, TEST, TRAIN), index=ids.index)


def save_split(split: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(path, index=False)


def load_or_create_split(
    path: Path,
    ids: pd.Series,
    make: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    """Reuse the frozen split if it covers the same ids, otherwise create it."""
    if path.exists():
        split = pd.read_parquet(path)
        if len(split) == len(ids) and set(split["id"]) == set(ids):
            return split
    split = make()
    save_split(split, path)
    return split
