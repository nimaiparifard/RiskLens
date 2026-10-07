"""The deployable scoring bundle.

Training produces ONE object that contains everything needed to go from raw
input to a decision: feature functions -> encoder -> selected features ->
model -> calibrator -> threshold / grades -> SHAP reasons. Serving loads it
from the MLflow registry and calls ``score``. Same code path, no skew.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from risklens.evaluation.cost import assign_grades
from risklens.explain.reasons import ReasonCoder
from risklens.explain.shap_explainer import make_explainer
from risklens.features.encoding import TabularEncoder
from risklens.interfaces import Classifier, ScoreResult
from risklens.models.calibration import Calibrator

Explain = bool | Literal["flagged"]

DECISIONS = {
    "credit": ("approve", "review", "reject"),
    "fraud": ("approve", "review", "block"),
}


def prepare_input(task: str, raw: pd.DataFrame) -> pd.DataFrame:
    if task == "credit":
        from risklens.features.registry import prepare_credit_input

        return prepare_credit_input(raw)
    from risklens.features.fraud import fraud_features

    return fraud_features(raw)


def pd_to_points(
    p: np.ndarray, pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0
) -> np.ndarray:
    """Scorecard scaling: 600 points at 50:1 good:bad odds, +20 points doubles the odds."""
    p = np.clip(np.asarray(p, dtype="float64"), 1e-6, 1 - 1e-6)
    odds_good = (1 - p) / p
    factor = pdo / np.log(2)
    return np.round(base_score + factor * (np.log(odds_good) - np.log(base_odds))).astype(int)


@dataclass
class Scorer:
    task: str
    encoder: TabularEncoder
    feature_names: list[str]
    model: Classifier
    calibrator: Calibrator
    threshold: float
    grade_bands: dict[str, float]
    explain_model: str = "lightgbm"
    top_k: int = 5
    review_band: float = 0.25
    input_columns: list[str] = field(default_factory=list)
    history_columns: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    reference: pd.DataFrame | None = None
    version: str = "local"
    reason_coder: ReasonCoder = field(default_factory=ReasonCoder)

    # ------------------------------------------------------------ pipeline
    def prepare(self, raw: pd.DataFrame) -> pd.DataFrame:
        return prepare_input(self.task, raw)

    def transform_prepared(self, prepared: pd.DataFrame) -> pd.DataFrame:
        return self.encoder.transform(prepared, columns=self.feature_names)

    def predict_proba_prepared(self, prepared: pd.DataFrame) -> np.ndarray:
        X = self.transform_prepared(prepared)
        return self.calibrator.transform(self.model.predict_proba(X))

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        return self.predict_proba_prepared(self.prepare(raw))

    def decide(self, p: np.ndarray) -> np.ndarray:
        ok, review, bad = DECISIONS[self.task]
        p = np.asarray(p)
        lower = self.threshold * (1 - self.review_band)
        return np.where(p >= self.threshold, bad, np.where(p >= lower, review, ok))

    # -------------------------------------------------------------- scoring
    def score_prepared(self, prepared: pd.DataFrame, explain: Explain = True) -> list[ScoreResult]:
        """``explain``: True = SHAP reasons for every row, False = none,
        "flagged" = only rows not approved (TreeSHAP is the expensive step; on the
        fraud path ~97% of transactions are approved and nobody reads their reasons)."""
        X = self.transform_prepared(prepared)
        p = self.calibrator.transform(self.model.predict_proba(X))
        grades = assign_grades(p, self.grade_bands)
        decisions = self.decide(p)
        points: list[int | None] = (
            [int(v) for v in pd_to_points(p)] if self.task == "credit" else [None] * len(p)
        )
        reasons: list[list] = [[] for _ in range(len(p))]
        ok = DECISIONS[self.task][0]
        rows = (
            [i for i in range(len(p)) if decisions[i] != ok]
            if explain == "flagged"
            else (list(range(len(p))) if explain else [])
        )
        if rows:
            expl = make_explainer(self.model, self.explain_model).explain(X.iloc[rows])
            for j, i in enumerate(rows):
                reasons[i] = self.reason_coder.top_reasons(
                    expl.values[j], expl.feature_names, prepared.iloc[i], k=self.top_k
                )
        return [
            ScoreResult(
                probability=float(p[i]),
                grade=str(grades[i]),
                decision=str(decisions[i]),
                threshold=float(self.threshold),
                reasons=reasons[i],
                model_version=self.version,
                points=points[i],
            )
            for i in range(len(p))
        ]

    def score(self, raw: pd.DataFrame, explain: Explain = True) -> list[ScoreResult]:
        return self.score_prepared(self.prepare(raw), explain=explain)

    def timed_score(
        self, raw: pd.DataFrame, explain: Explain = True
    ) -> tuple[list[ScoreResult], float]:
        t0 = time.perf_counter()
        res = self.score(raw, explain)
        return res, (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------ metadata
    def info(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "version": self.version,
            "threshold": self.threshold,
            "grade_bands": self.grade_bands,
            "n_features": len(self.feature_names),
            "model": type(self.model).__name__,
            "metrics": self.metadata.get("metrics", {}),
            "trained_at": self.metadata.get("trained_at"),
            "git_sha": self.metadata.get("git_sha"),
            "input_columns": self.input_columns,
            "history_columns": self.history_columns,
        }

    def reference_records(self, n: int | None = None) -> list[dict[str, Any]]:
        if self.reference is None:
            return []
        ref = self.reference if n is None else self.reference.head(n)
        return ref.replace({np.nan: None}).to_dict(orient="records")

    # ------------------------------------------------------------- storage
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: Path) -> Scorer:
        obj = joblib.load(path)
        if not isinstance(obj, Scorer):
            raise TypeError(f"{path} does not contain a Scorer")
        return obj
