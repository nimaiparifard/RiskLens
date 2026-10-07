"""Data & prediction drift.

PSI (Population Stability Index) is the banking standard:
  PSI < 0.1  stable | 0.1 - 0.2 moderate shift | > 0.2 significant -> investigate / retrain
Evidently is used (optionally) to render a full HTML report for humans.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PSI_SIGNIFICANT = 0.2
PSI_MODERATE = 0.1
_EPS = 1e-4


def psi_numeric(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    e = np.asarray(expected, dtype="float64")
    a = np.asarray(actual, dtype="float64")
    e_fin, a_fin = e[np.isfinite(e)], a[np.isfinite(a)]
    if len(e_fin) == 0 or len(a_fin) == 0:
        return 0.0
    edges = np.unique(np.quantile(e_fin, np.linspace(0, 1, bins + 1)[1:-1]))
    e_counts = np.bincount(np.digitize(e_fin, edges), minlength=len(edges) + 1).astype(float)
    a_counts = np.bincount(np.digitize(a_fin, edges), minlength=len(edges) + 1).astype(float)
    # missing values are their own bin
    e_counts = np.r_[e_counts, (~np.isfinite(e)).sum()]
    a_counts = np.r_[a_counts, (~np.isfinite(a)).sum()]
    return _psi(e_counts, a_counts)


def psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    e = expected.astype("object").fillna("__nan__").astype(str).value_counts()
    a = actual.astype("object").fillna("__nan__").astype(str).value_counts()
    cats = e.index.union(a.index)
    return _psi(
        e.reindex(cats, fill_value=0).to_numpy(float), a.reindex(cats, fill_value=0).to_numpy(float)
    )


def _psi(e_counts: np.ndarray, a_counts: np.ndarray) -> float:
    pe = np.clip(e_counts / max(e_counts.sum(), 1), _EPS, None)
    pa = np.clip(a_counts / max(a_counts.sum(), 1), _EPS, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


@dataclass
class DriftResult:
    feature_psi: dict[str, float]
    prediction_psi: float | None
    n_reference: int
    n_current: int
    drifted_features: list[str] = field(default_factory=list)
    threshold: float = PSI_SIGNIFICANT

    @property
    def max_psi(self) -> float:
        vals = list(self.feature_psi.values()) + (
            [self.prediction_psi] if self.prediction_psi is not None else []
        )
        return float(max(vals)) if vals else 0.0

    @property
    def drift_detected(self) -> bool:
        return self.max_psi > self.threshold

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.update(max_psi=self.max_psi, drift_detected=self.drift_detected)
        return d


def compute_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str] | None = None,
    score_column: str | None = "score",
    threshold: float = PSI_SIGNIFICANT,
) -> DriftResult:
    cols = features or [c for c in reference.columns if c != score_column]
    cols = [c for c in cols if c in current.columns]
    psi: dict[str, float] = {}
    for c in cols:
        ref, cur = reference[c], current[c]
        numeric = pd.api.types.is_numeric_dtype(ref) and pd.api.types.is_numeric_dtype(
            pd.to_numeric(cur, errors="coerce")
        )
        if numeric:
            psi[c] = round(
                psi_numeric(ref.to_numpy(), pd.to_numeric(cur, errors="coerce").to_numpy()), 5
            )
        else:
            psi[c] = round(psi_categorical(ref, cur), 5)
    pred_psi = None
    if score_column and score_column in reference and score_column in current:
        pred_psi = round(
            psi_numeric(reference[score_column].to_numpy(), current[score_column].to_numpy()), 5
        )
    return DriftResult(
        feature_psi=psi,
        prediction_psi=pred_psi,
        n_reference=len(reference),
        n_current=len(current),
        drifted_features=sorted(
            [c for c, v in psi.items() if v > threshold], key=lambda c: -psi[c]
        ),
        threshold=threshold,
    )


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, path: Path) -> Path | None:
    """Render an Evidently data-drift HTML report (supports Evidently >= 0.7 and legacy API)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    common = [c for c in reference.columns if c in current.columns]
    ref, cur = reference[common], current[common]
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset

        snapshot = Report([DataDriftPreset()]).run(reference_data=ref, current_data=cur)
        snapshot.save_html(str(path))
        return path
    except ImportError:
        pass
    try:
        from evidently.metric_preset import DataDriftPreset as LegacyPreset
        from evidently.report import Report as LegacyReport

        report = LegacyReport(metrics=[LegacyPreset()])
        report.run(reference_data=ref, current_data=cur)
        report.save_html(str(path))
        return path
    except ImportError:
        return None
