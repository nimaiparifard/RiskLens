"""XGBoost classifier (hist tree method)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from risklens.interfaces import Classifier
from risklens.registry import register_model


@register_model("xgboost")
class XGBModel(Classifier):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> XGBModel:
        import xgboost as xgb

        params: dict[str, Any] = dict(self.params)
        es = params.pop("early_stopping_rounds", None)
        pos_weight = params.pop("pos_weight", None)
        if pos_weight:
            params["scale_pos_weight"] = pos_weight
        params.setdefault("tree_method", "hist")
        params.setdefault("eval_metric", "auc")
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", -1)
        has_valid = X_valid is not None and y_valid is not None
        if es and has_valid:
            params["early_stopping_rounds"] = es
        self.model_ = xgb.XGBClassifier(**params)
        fit_kw: dict[str, Any] = {"verbose": False}
        if has_valid:
            fit_kw["eval_set"] = [(X_valid, y_valid)]
        self.model_.fit(X, y, **fit_kw)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.feature_names_])[:, 1]

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        import xgboost as xgb

        booster = self.model_.get_booster()
        best = getattr(self.model_, "best_iteration", None)
        rng = (0, best + 1) if best is not None else (0, 0)
        return booster.predict(
            xgb.DMatrix(X[self.feature_names_]), pred_contribs=True, iteration_range=rng
        )

    def feature_importance(self) -> pd.Series:
        score = self.model_.get_booster().get_score(importance_type="total_gain")
        return pd.Series(score).reindex(self.feature_names_).fillna(0).sort_values(ascending=False)
