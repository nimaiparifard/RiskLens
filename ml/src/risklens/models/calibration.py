"""Probability calibration fitted on the validation set (never on train)."""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from risklens.models.base import logit


class Calibrator:
    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "sigmoid", "none"):
            raise ValueError(method)
        self.method = method

    def fit(self, p: np.ndarray, y: np.ndarray) -> Calibrator:
        p, y = np.asarray(p, dtype="float64"), np.asarray(y)
        if self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(
                p, y
            )
        elif self.method == "sigmoid":  # Platt scaling on the logit
            self.model_ = LogisticRegression(C=1e4, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype="float64")
        if self.method == "isotonic":
            return self.model_.predict(p)
        if self.method == "sigmoid":
            return self.model_.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        return p
