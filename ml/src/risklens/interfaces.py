"""Interfaces (Rule 2: program to interfaces).

Every model, explainer and feature builder implements one of these, so the
pipelines never depend on a concrete library. Adding XGBoost, a new data
source or a new explainer = add a class + register its name.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class Classifier(ABC):
    """Binary classifier returning P(y=1)."""

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)
        self.feature_names_: list[str] = []

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> Classifier: ...

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return a 1-d array of positive-class probabilities."""

    def get_params(self) -> dict[str, Any]:
        return dict(self.params)

    def feature_importance(self) -> pd.Series | None:
        """Importance per feature, if the model provides one."""
        return None

    def clone(self) -> Classifier:
        return type(self)(**self.params)


@dataclass
class Explanation:
    """Per-row additive contributions (SHAP values in log-odds space)."""

    values: np.ndarray  # (n_rows, n_features)
    base_value: float
    feature_names: list[str]
    data: pd.DataFrame | None = None


@runtime_checkable
class Explainer(Protocol):
    def explain(self, X: pd.DataFrame) -> Explanation: ...


@runtime_checkable
class FeatureBuilder(Protocol):
    """Builds per-entity features from one source table.

    ``build`` must be pure: DataFrame(s) in -> DataFrame out, indexed by the id.
    """

    name: str

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame: ...


@runtime_checkable
class HistoryStore(Protocol):
    """Online store of recent events per entity (used by fraud velocity features)."""

    def get(self, key: str) -> list[dict[str, Any]]: ...

    def append(self, key: str, event: dict[str, Any]) -> None: ...


@dataclass
class Reason:
    feature: str
    value: Any
    contribution: float
    direction: str  # "increases_risk" | "decreases_risk"
    text: str


@dataclass
class ScoreResult:
    probability: float
    grade: str | None
    decision: str
    threshold: float
    reasons: list[Reason] = field(default_factory=list)
    model_version: str = "unknown"
    points: int | None = None  # scorecard-style score (higher = safer)
