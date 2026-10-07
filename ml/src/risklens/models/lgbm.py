"""LightGBM classifier (optionally with a focal-loss objective)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from risklens.interfaces import Classifier
from risklens.models.base import focal_loss_objective, sigmoid
from risklens.registry import register_model


def eval_kwargs(X_valid: pd.DataFrame, y_valid: pd.Series) -> dict[str, Any]:
    """LightGBM >= 4.7 renamed ``eval_set`` to ``eval_X`` / ``eval_y``."""
    import inspect

    import lightgbm as lgb

    if "eval_X" in inspect.signature(lgb.LGBMClassifier.fit).parameters:
        return {"eval_X": (X_valid,), "eval_y": (y_valid,)}
    return {"eval_set": [(X_valid, y_valid)]}


@register_model("lightgbm")
class LGBMModel(Classifier):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> LGBMModel:
        import lightgbm as lgb

        params: dict[str, Any] = dict(self.params)
        es = params.pop("early_stopping_rounds", None)
        pos_weight = params.pop("pos_weight", None)
        gamma = params.pop("focal_gamma", None)
        alpha = params.pop("focal_alpha", 0.25)
        params.setdefault("random_state", 42)
        params.setdefault("verbose", -1)
        params.setdefault("n_jobs", -1)
        # early stopping must watch AUC only: with scale_pos_weight the default
        # binary_logloss worsens from the first trees and would stop training early
        params.setdefault("metric", "auc")
        if pos_weight and not gamma:
            params["scale_pos_weight"] = pos_weight
        self._focal = bool(gamma)
        if self._focal:
            params["objective"] = focal_loss_objective(gamma, alpha)

        self.model_ = lgb.LGBMClassifier(**params)
        fit_kw: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kw.update(eval_kwargs(X_valid, y_valid))
            fit_kw["eval_metric"] = "auc"
            if es:
                fit_kw["callbacks"] = [lgb.early_stopping(es, verbose=False)]
        self.model_.fit(X, y, **fit_kw)
        self.feature_names_ = list(X.columns)
        self.best_iteration_ = int(self.model_.best_iteration_ or params.get("n_estimators", 100))
        return self

    @property
    def booster(self) -> Any:
        return self.model_.booster_

    @staticmethod
    def _threads(X: pd.DataFrame) -> dict[str, int]:
        # single-row serving: OpenMP fan-out costs more than it saves
        return {"num_threads": 1} if len(X) <= 256 else {}

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.feature_names_]
        if self._focal:
            raw = self.booster.predict(
                X, raw_score=True, num_iteration=self.best_iteration_, **self._threads(X)
            )
            return sigmoid(raw)
        return self.booster.predict(X, num_iteration=self.best_iteration_, **self._threads(X))

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        """Exact TreeSHAP values (log-odds), last column = expected value."""
        return self.booster.predict(
            X[self.feature_names_],
            pred_contrib=True,
            num_iteration=self.best_iteration_,
            # TreeSHAP is ~100x heavier than predict: let it use all cores
        )

    def feature_importance(self) -> pd.Series:
        imp = self.booster.feature_importance(importance_type="gain")
        return pd.Series(imp, index=self.feature_names_).sort_values(ascending=False)
