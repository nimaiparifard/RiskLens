"""Fairness audit of DECISIONS (not just scores) by protected attribute.

Uses Fairlearn's MetricFrame when available (same definitions otherwise):
  demographic parity difference = max - min approval rate across groups
  equalized odds difference     = max(range of TPR, range of FPR)
Optional mitigation: Fairlearn ThresholdOptimizer (group-specific thresholds)
and the resulting accuracy / cost trade-off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FairnessResult:
    by_feature: dict[str, pd.DataFrame] = field(default_factory=dict)
    gaps: dict[str, dict[str, float]] = field(default_factory=dict)
    mitigation: dict[str, Any] = field(default_factory=dict)
    gate_features: list[str] | None = None

    @property
    def max_gap(self) -> float:
        """Max demographic-parity gap over the attributes that gate promotion."""
        gated = self.gate_features if self.gate_features is not None else list(self.gaps)
        vals = [g["demographic_parity_diff"] for f, g in self.gaps.items() if f in gated]
        return float(max(vals)) if vals else 0.0

    def metrics(self) -> dict[str, float]:
        out = {}
        for feat, g in self.gaps.items():
            for k, v in g.items():
                out[f"fairness_{feat.lower()}_{k}"] = v
        out["fairness_max_dp_gap"] = self.max_gap
        return out

    def to_markdown(self) -> str:
        lines = ["# Fairness report", ""]
        lines.append("Decisions audited at the cost-optimal threshold. Protected attributes are")
        lines.append(
            "**excluded from the model's inputs**; this audit checks for indirect (proxy) bias."
        )
        lines.append("")
        for feat, table in self.by_feature.items():
            lines.append(f"## {feat}")
            lines.append("")
            from risklens.evaluation.report import df_to_markdown

            lines.append(df_to_markdown(table))
            lines.append("")
            g = self.gaps[feat]
            lines.append(
                f"- Demographic parity difference (approval rate): **{g['demographic_parity_diff']:.4f}**"
            )
            lines.append(f"- Equalized odds difference: **{g['equalized_odds_diff']:.4f}**")
            gated = self.gate_features is None or feat in self.gate_features
            lines.append(f"- Promotion gate: **{'yes' if gated else 'no (reported only)'}**")
            lines.append("")
        if self.mitigation:
            lines.append("## Mitigation: Fairlearn ThresholdOptimizer (demographic parity)")
            lines.append("")
            lines.append(
                "Group-specific thresholds chosen to equalise approval rates - and what that costs."
            )
            lines.append("")
            for feat, m in self.mitigation.items():
                lines.append(f"- **{feat}**: " + ", ".join(f"{k} = {v}" for k, v in m.items()))
        return "\n".join(lines)


def group_table(y: np.ndarray, flagged: np.ndarray, groups: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"g": groups.fillna("missing").astype(str).to_numpy(), "y": y, "f": flagged})
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d["y"] == 1, d["y"] == 0
        rows.append(
            {
                "group": g,
                "n": len(d),
                "default_rate": d["y"].mean(),
                "approval_rate": 1 - d["f"].mean(),
                "tpr": d.loc[pos, "f"].mean()
                if pos.any()
                else np.nan,  # defaulters correctly rejected
                "fpr": d.loc[neg, "f"].mean()
                if neg.any()
                else np.nan,  # good customers wrongly rejected
            }
        )
    return pd.DataFrame(rows)


def fairness_report(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
    sensitive: pd.DataFrame,
    min_group_size: int = 30,
    mitigate: bool = False,
    gate_features: list[str] | None = None,
) -> FairnessResult:
    y = np.asarray(y).astype(int)
    flagged = (np.asarray(p) >= threshold).astype(int)
    res = FairnessResult(gate_features=gate_features)
    for feat in sensitive.columns:
        table = group_table(y, flagged, sensitive[feat])
        res.by_feature[feat] = table
        t = table[table["n"] >= min_group_size]
        res.gaps[feat] = {
            "demographic_parity_diff": float(t["approval_rate"].max() - t["approval_rate"].min())
            if len(t)
            else 0.0,
            "equalized_odds_diff": float(
                max(t["tpr"].max() - t["tpr"].min(), t["fpr"].max() - t["fpr"].min())
            )
            if len(t)
            else 0.0,
        }
    if mitigate:
        for feat in sensitive.columns:
            g = sensitive[feat].fillna("missing").astype(str)
            keep = (
                g.map(g.value_counts()).to_numpy() >= min_group_size
            )  # tiny groups can't be fitted
            try:  # optional analysis: must never break a training run
                res.mitigation[feat] = _threshold_optimizer(
                    y[keep], np.asarray(p)[keep], flagged[keep], g[keep].reset_index(drop=True)
                )
            except Exception as exc:
                res.mitigation[feat] = {"status": f"failed: {exc}"}
    return res


def _gap(y: np.ndarray, flagged: np.ndarray, groups: pd.Series) -> float:
    table = group_table(y, flagged, groups)
    return float(table["approval_rate"].max() - table["approval_rate"].min())


def _threshold_optimizer(
    y: np.ndarray, p: np.ndarray, flagged: np.ndarray, groups: pd.Series
) -> dict[str, Any]:
    """Group-specific thresholds (demographic parity) and what they would cost."""
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
    except ImportError:
        return {"status": "fairlearn not installed"}
    from sklearn.base import BaseEstimator, ClassifierMixin
    from sklearn.metrics import balanced_accuracy_score

    class _Prefit(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
        """Scores are precomputed: the 'model' just returns its single input column."""

        def fit(self, X: Any, y: Any) -> _Prefit:
            self.classes_ = np.array([0, 1])
            return self

        def predict(self, X: Any) -> np.ndarray:
            return np.asarray(X)[:, 0]

    X = p.reshape(-1, 1)
    g = groups.fillna("missing").astype(str)
    opt = ThresholdOptimizer(
        estimator=_Prefit().fit(X, y),
        constraints="demographic_parity",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict",
    )
    opt.fit(X, y, sensitive_features=g.to_numpy())
    mitigated = np.asarray(opt.predict(X, sensitive_features=g.to_numpy(), random_state=0)).astype(
        int
    )
    return {
        "approval_rate_gap_before": round(_gap(y, flagged, g), 4),
        "approval_rate_gap_after": round(_gap(y, mitigated, g), 4),
        "balanced_accuracy_before": round(float(balanced_accuracy_score(y, flagged)), 4),
        "balanced_accuracy_after": round(float(balanced_accuracy_score(y, mitigated)), 4),
        "approval_rate_after": round(float(1 - mitigated.mean()), 4),
    }
