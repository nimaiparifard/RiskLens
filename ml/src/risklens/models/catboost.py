"""CatBoost classifier (ordered boosting; robust defaults on tabular data)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from risklens.interfaces import Classifier
from risklens.registry import register_model


@register_model("catboost")
class CatBoostModel(Classifier):
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> CatBoostModel:
        from catboost import CatBoostClassifier

        params: dict[str, Any] = dict(self.params)
        es = params.pop("early_stopping_rounds", None)
        pos_weight = params.pop("pos_weight", None)
        if pos_weight:
            params["scale_pos_weight"] = pos_weight
        params.setdefault("random_seed", params.pop("random_state", 42))
        params.setdefault("eval_metric", "AUC")
        params.setdefault("verbose", False)
        params.setdefault("allow_writing_files", False)
        params.setdefault("thread_count", -1)
        self.model_ = CatBoostClassifier(**params)
        fit_kw: dict[str, Any] = {}
        if X_valid is not None and y_valid is not None:
            fit_kw["eval_set"] = (X_valid, y_valid)
            if es:
                fit_kw["early_stopping_rounds"] = es
                fit_kw["use_best_model"] = True
        self.model_.fit(X, y, **fit_kw)
        self.feature_names_ = list(X.columns)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.feature_names_])[:, 1]

    def contributions(self, X: pd.DataFrame) -> np.ndarray:
        from catboost import Pool

        return self.model_.get_feature_importance(Pool(X[self.feature_names_]), type="ShapValues")

    def feature_importance(self) -> pd.Series:
        return pd.Series(
            self.model_.get_feature_importance(), index=self.feature_names_
        ).sort_values(ascending=False)
