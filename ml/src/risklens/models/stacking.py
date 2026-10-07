"""Stacking ensemble: out-of-fold base predictions -> logistic meta-model."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from risklens.interfaces import Classifier
from risklens.models.base import inner_split, logit
from risklens.registry import create_model, register_model

log = logging.getLogger(__name__)


@register_model("stacking")
class StackingModel(Classifier):
    """params:
    base_models: list of registered model names
    base_params: {name: params} (injected by the pipeline from the config)
    n_folds:     CV folds for out-of-fold predictions
    cv:          "stratified" | "time"
    """

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._prefit: dict[str, Classifier] = {}

    def set_prefit(self, models: dict[str, Classifier]) -> StackingModel:
        """Reuse base models already fitted on the full training set."""
        self._prefit = dict(models)
        return self

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> StackingModel:
        names: list[str] = list(self.params["base_models"])
        base_params: dict[str, dict] = self.params.get("base_params", {})
        n_folds = int(self.params.get("n_folds", 5))
        seed = int(self.params.get("seed", 42))
        y_arr = np.asarray(y)
        if self.params.get("cv", "stratified") == "time":
            splits = list(TimeSeriesSplit(n_splits=n_folds).split(X))
        else:
            splits = list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(X, y_arr))

        oof = np.full((len(X), len(names)), np.nan)
        for j, name in enumerate(names):
            for k, (tr, va) in enumerate(splits):
                tr_in, es = inner_split(pd.Series(y_arr[tr]), 0.1, seed + k)
                m = create_model(name, **base_params.get(name, {}))
                Xtr = X.iloc[tr]
                m.fit(
                    Xtr.iloc[tr_in],
                    pd.Series(y_arr[tr][tr_in]),
                    Xtr.iloc[es],
                    pd.Series(y_arr[tr][es]),
                )
                oof[va, j] = m.predict_proba(X.iloc[va])
            log.info("stacking: OOF predictions for %s done", name)

        mask = ~np.isnan(oof).any(axis=1)
        self.meta_ = LogisticRegression(C=float(self.params.get("meta_C", 1.0)), max_iter=1000)
        self.meta_.fit(logit(oof[mask]), y_arr[mask])
        self.oof_ = oof

        self.base_models_: dict[str, Classifier] = {}
        for name in names:
            if name in self._prefit:
                self.base_models_[name] = self._prefit[name]
            else:
                m = create_model(name, **base_params.get(name, {}))
                self.base_models_[name] = m.fit(X, y, X_valid, y_valid)
        self._prefit = {}
        self.names_ = names
        self.feature_names_ = list(X.columns)
        return self

    def base_predictions(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.base_models_[n].predict_proba(X) for n in self.names_])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.meta_.predict_proba(logit(self.base_predictions(X)))[:, 1]

    def meta_weights(self) -> dict[str, float]:
        return dict(zip(self.names_, self.meta_.coef_[0].round(4).tolist(), strict=True))

    def feature_importance(self) -> pd.Series | None:
        for name in ("lightgbm", "xgboost", "catboost"):
            if name in getattr(self, "base_models_", {}):
                return self.base_models_[name].feature_importance()
        return None
