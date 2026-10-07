"""Logistic regression on Weight-of-Evidence bins: the classic bank scorecard.

Every feature is quantile-binned (missing = its own bin), each bin is replaced
by its WOE = ln(%bad / %good), features are ranked by Information Value and
the top ones feed a logistic regression. The result converts to a points
table (``scorecard()``) that a credit committee can read line by line.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from risklens.interfaces import Classifier
from risklens.registry import register_model


class WOEBinner:
    def __init__(self, max_bins: int = 10, smoothing: float = 0.5) -> None:
        self.max_bins = max_bins
        self.smoothing = smoothing

    def fit(self, X: pd.DataFrame, y: pd.Series) -> WOEBinner:
        y_arr = np.asarray(y, dtype="float64")
        n_bad, n_good = y_arr.sum(), len(y_arr) - y_arr.sum()
        self.edges_: dict[str, np.ndarray] = {}
        self.woe_: dict[str, np.ndarray] = {}
        iv = {}
        for c in X.columns:
            x = X[c].to_numpy(dtype="float64")
            finite = x[~np.isnan(x)]
            if finite.size == 0:
                continue
            qs = np.unique(np.quantile(finite, np.linspace(0, 1, self.max_bins + 1)[1:-1]))
            self.edges_[c] = qs
            bins = self._bin(x, qs)
            n_bins = len(qs) + 2  # +1 for right-most bin, +1 for missing
            bad = np.bincount(bins, weights=y_arr, minlength=n_bins)
            tot = np.bincount(bins, minlength=n_bins)
            good = tot - bad
            pb = (bad + self.smoothing) / (n_bad + self.smoothing * n_bins)
            pg = (good + self.smoothing) / (n_good + self.smoothing * n_bins)
            woe = np.log(pb / pg)
            self.woe_[c] = woe
            iv[c] = float(np.sum((pb - pg) * woe))
        self.iv_ = pd.Series(iv).sort_values(ascending=False)
        return self

    @staticmethod
    def _bin(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        bins = np.digitize(x, edges)  # 0..len(edges)
        bins[np.isnan(x)] = len(edges) + 1
        return bins

    def transform(self, X: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        out = {}
        for c in columns:
            x = X[c].to_numpy(dtype="float64") if c in X else np.full(len(X), np.nan)
            out[c] = self.woe_[c][self._bin(x, self.edges_[c])]
        return pd.DataFrame(out, index=X.index)


@register_model("logreg_woe")
class WOELogisticRegression(Classifier):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> WOELogisticRegression:
        p: dict[str, Any] = dict(self.params)
        self.binner_ = WOEBinner(max_bins=p.get("max_bins", 10)).fit(X, y)
        iv = self.binner_.iv_
        keep = iv[iv >= p.get("min_iv", 0.02)].index[: p.get("max_features", 60)]
        if len(keep) == 0:
            keep = iv.index[: p.get("max_features", 60)]
        self.columns_ = list(keep)
        W = self.binner_.transform(X, self.columns_)
        pos_weight = p.get("pos_weight")
        self.model_ = LogisticRegression(
            C=p.get("C", 0.1),
            class_weight={0: 1.0, 1: pos_weight} if pos_weight else None,
            max_iter=2000,
        )
        self.model_.fit(W, y)
        self.woe_mean_ = W.mean().to_numpy()
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(self.binner_.transform(X, self.columns_))[:, 1]

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Linear SHAP: coef * (woe - mean woe); last column = expected logit."""
        W = self.binner_.transform(X, self.columns_).to_numpy()
        coef = self.model_.coef_[0]
        contrib = (W - self.woe_mean_) * coef
        full = np.zeros((len(X), len(self.feature_names_) + 1))
        pos = {c: i for i, c in enumerate(self.feature_names_)}
        for j, c in enumerate(self.columns_):
            full[:, pos[c]] = contrib[:, j]
        full[:, -1] = self.model_.intercept_[0] + float(self.woe_mean_ @ coef)
        return full

    def feature_importance(self) -> pd.Series:
        return pd.Series(np.abs(self.model_.coef_[0]), index=self.columns_).sort_values(
            ascending=False
        )

    def scorecard(
        self, pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0
    ) -> pd.DataFrame:
        """Points table: score = offset - factor * ln(odds of default)."""
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        n = len(self.columns_)
        intercept = self.model_.intercept_[0]
        rows = []
        for c, b in zip(self.columns_, self.model_.coef_[0], strict=True):
            edges = self.binner_.edges_[c]
            labels = [f"< {edges[0]:.4g}"] if len(edges) else []
            labels += [f"[{lo:.4g}, {hi:.4g})" for lo, hi in itertools.pairwise(edges)]
            labels += [f">= {edges[-1]:.4g}" if len(edges) else "all", "missing"]
            for label, woe in zip(labels, self.binner_.woe_[c], strict=True):
                points = -(b * woe + intercept / n) * factor + offset / n
                rows.append({"feature": c, "bin": label, "woe": woe, "points": round(points, 1)})
        return pd.DataFrame(rows)
