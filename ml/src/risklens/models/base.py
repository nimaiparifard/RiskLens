"""Shared model utilities: imbalance strategies, focal loss, logit helpers."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

EPS = 1e-7


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype="float64"), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def positive_weight(y: pd.Series | np.ndarray) -> float:
    y_arr = np.asarray(y)
    pos = float(y_arr.mean())
    return (1.0 - pos) / max(pos, 1e-6)


def apply_imbalance(
    strategy: str, X: pd.DataFrame, y: pd.Series, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Return (X, y, extra model params) for the chosen imbalance strategy.

    * scale_pos_weight / class_weight -> models receive ``pos_weight``
    * undersample -> random majority undersampling to 1:3
    * smote       -> SMOTE oversampling (NaNs median-imputed; GBDTs rarely benefit)
    """
    if strategy in ("none", None):
        return X, y, {}
    if strategy in ("scale_pos_weight", "class_weight"):
        return X, y, {"pos_weight": positive_weight(y)}
    if strategy == "undersample":
        rng = np.random.default_rng(seed)
        pos_idx = np.flatnonzero(np.asarray(y) == 1)
        neg_idx = np.flatnonzero(np.asarray(y) == 0)
        keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), 3 * len(pos_idx)), replace=False)
        idx = np.sort(np.r_[pos_idx, keep_neg])
        return X.iloc[idx], y.iloc[idx], {}
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        filled = X.fillna(X.median())
        Xs, ys = SMOTE(random_state=seed, sampling_strategy=0.3).fit_resample(filled, y)
        return pd.DataFrame(Xs, columns=X.columns), pd.Series(ys, name=y.name), {}
    raise ValueError(f"unknown imbalance strategy {strategy}")


def focal_loss_objective(
    gamma: float = 2.0, alpha: float = 0.25
) -> Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Custom LightGBM objective: focal loss (Lin et al. 2017) on raw margins.

    Gradient / hessian via central finite differences of the closed-form loss
    (exact enough and much less error-prone than the analytic second derivative).
    """

    def loss(z: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = np.clip(sigmoid(z), EPS, 1 - EPS)
        return -(
            y * alpha * (1 - p) ** gamma * np.log(p)
            + (1 - y) * (1 - alpha) * p**gamma * np.log(1 - p)
        )

    def objective(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = 1e-4
        lp, l0, lm = loss(y_pred + h, y_true), loss(y_pred, y_true), loss(y_pred - h, y_true)
        grad = (lp - lm) / (2 * h)
        hess = np.maximum((lp - 2 * l0 + lm) / h**2, 1e-6)
        return grad, hess

    return objective


def inner_split(y: pd.Series, frac: float = 0.1, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Stratified train / early-stopping split inside a CV fold."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    tr, es = train_test_split(idx, test_size=frac, stratify=np.asarray(y), random_state=seed)
    return tr, es
