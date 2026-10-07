"""Evaluation report: metrics + business numbers + plots for one scorer on one dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from risklens.config import CostConfig
from risklens.evaluation import plots
from risklens.evaluation.cost import (
    amounts_for,
    approval_rate_at_risk,
    business_summary,
    grade_table,
)
from risklens.evaluation.metrics import classification_metrics, confusion_at


def df_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Tiny markdown renderer (avoids the optional ``tabulate`` dependency)."""
    cols = [str(c) for c in df.columns]

    def fmt(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines)


@dataclass
class EvaluationReport:
    metrics: dict[str, float]
    grades: pd.DataFrame
    confusion: dict[str, int]
    plots: dict[str, Path] = field(default_factory=dict)

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metrics": self.metrics,
            "confusion": self.confusion,
            "grades": self.grades.to_dict(orient="records"),
        }
        path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
        return path


def evaluate_predictions(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    costs: CostConfig,
    frame: pd.DataFrame | None = None,
    out_dir: Path | None = None,
    prefix: str = "test",
) -> EvaluationReport:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype="float64")
    amounts = amounts_for(frame, costs, len(y))
    metrics = classification_metrics(y, p)
    metrics.update(business_summary(y, p, threshold, amounts, costs))
    metrics["approval_rate_at_bad_rate_5pct"] = approval_rate_at_risk(y, p, 0.05)
    conf = confusion_at(y, p, threshold)
    grades = grade_table(y, p, costs.grade_bands)
    report = EvaluationReport(metrics=metrics, grades=grades, confusion=conf)
    if out_dir is not None:
        out_dir = Path(out_dir)
        report.plots = {
            "roc": plots.roc_plot({prefix: (y, p)}, out_dir / f"{prefix}_roc.png"),
            "pr": plots.pr_plot({prefix: (y, p)}, out_dir / f"{prefix}_pr.png"),
            "calibration": plots.calibration_plot(
                y, {prefix: p}, out_dir / f"{prefix}_calibration.png"
            ),
            "confusion": plots.confusion_plot(conf, threshold, out_dir / f"{prefix}_confusion.png"),
            "distribution": plots.score_distribution_plot(
                y, p, threshold, out_dir / f"{prefix}_scores.png"
            ),
        }
    return report


RESULT_COLUMNS = ["roc_auc", "pr_auc", "ks", "gini", "brier", "expected_cost", "recall_at_1pct_fpr"]


def results_table(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(rows).T
    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    return df[cols].reset_index(names="model")
