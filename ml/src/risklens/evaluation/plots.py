"""Evaluation plots (matplotlib, headless). Each function returns the saved path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def roc_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.4f})", lw=1.6)
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right", fontsize=8)
    return _save(fig, path)


def pr_plot(curves: dict[str, tuple[np.ndarray, np.ndarray]], path: Path) -> Path:
    from sklearn.metrics import average_precision_score, precision_recall_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, p) in curves.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.4f})", lw=1.6)
    base = float(np.mean(next(iter(curves.values()))[0]))
    ax.axhline(base, ls="--", color="grey", lw=1)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    ax.legend(loc="upper right", fontsize=8)
    return _save(fig, path)


def calibration_plot(
    y: np.ndarray, probs: dict[str, np.ndarray], path: Path, n_bins: int = 15
) -> Path:
    from sklearn.calibration import calibration_curve

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    hi = 0.0
    for name, p in probs.items():
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, "o-", ms=3, label=name)
        hi = max(hi, mean.max(), frac.max())
    ax.plot([0, hi], [0, hi], "--", color="grey", lw=1)
    ax.set(
        xlabel="Mean predicted probability",
        ylabel="Observed default rate",
        title="Reliability diagram",
    )
    ax.legend(fontsize=8)
    return _save(fig, path)


def confusion_plot(conf: dict[str, int], threshold: float, path: Path) -> Path:
    plt = _plt()
    m = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:,}", ha="center", va="center", color="black")
    ax.set_xticks([0, 1], ["predicted good", "predicted bad"])
    ax.set_yticks([0, 1], ["actual good", "actual bad"])
    ax.set_title(f"Confusion matrix @ threshold {threshold:.3f}")
    return _save(fig, path)


def score_distribution_plot(y: np.ndarray, p: np.ndarray, threshold: float, path: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, np.quantile(p, 0.995), 60)
    ax.hist(p[np.asarray(y) == 0], bins=bins, alpha=0.6, density=True, label="good")
    ax.hist(p[np.asarray(y) == 1], bins=bins, alpha=0.6, density=True, label="bad")
    ax.axvline(threshold, color="black", ls="--", lw=1, label="threshold")
    ax.set(xlabel="Predicted probability", ylabel="density", title="Score distribution by class")
    ax.legend()
    return _save(fig, path)


def importance_plot(
    importance: pd.Series, path: Path, top: int = 25, title: str = "Mean |SHAP|"
) -> Path:
    plt = _plt()
    imp = importance.sort_values(ascending=False).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(imp) + 1))
    ax.barh(imp.index, imp.to_numpy())
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    return _save(fig, path)


def psi_plot(psi: pd.Series, path: Path, threshold: float = 0.2) -> Path:
    plt = _plt()
    s = psi.sort_values(ascending=False).head(25)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(s) + 1))
    ax.barh(s.index, s.to_numpy(), color=["#d62728" if v > threshold else "#1f77b4" for v in s])
    ax.axvline(threshold, color="black", ls="--", lw=1)
    ax.set_title("PSI per feature")
    return _save(fig, path)


def _save(fig: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    _plt().close(fig)
    return path
