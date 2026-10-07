"""Model-quality metrics. ROC-AUC is primary, but banks also need KS, Gini,
calibration (Brier) and operating-point metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov: max separation between bad and good score CDFs."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def recall_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float = 0.01) -> float:
    """Share of positives caught when only ``target_fpr`` of negatives are flagged."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.interp(target_fpr, fpr, tpr))


def threshold_at_fpr(y: np.ndarray, p: np.ndarray, target_fpr: float) -> float:
    neg = np.sort(np.asarray(p)[np.asarray(y) == 0])
    if len(neg) == 0:
        return 0.5
    return float(np.quantile(neg, 1.0 - target_fpr))


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.clip(np.asarray(p, dtype="float64"), 1e-7, 1 - 1e-7)
    auc = float(roc_auc_score(y, p))
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "pr_auc": float(average_precision_score(y, p)),
        "ks": ks_statistic(y, p),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p)),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "positive_rate": float(y.mean()),
        "mean_prediction": float(p.mean()),
    }


def confusion_at(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, int]:
    y = np.asarray(y).astype(bool)
    flag = np.asarray(p) >= threshold
    return {
        "tp": int((flag & y).sum()),
        "fp": int((flag & ~y).sum()),
        "tn": int((~flag & ~y).sum()),
        "fn": int((~flag & y).sum()),
    }
