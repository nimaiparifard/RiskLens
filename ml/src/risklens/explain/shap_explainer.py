"""SHAP explanations.

Tree models expose exact TreeSHAP natively (LightGBM ``pred_contrib``,
XGBoost ``pred_contribs``, CatBoost ``ShapValues``) - identical to
``shap.TreeExplainer`` but with no extra dependency and ~1 ms per row, so the
serving container can explain every single prediction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from risklens.interfaces import Classifier, Explanation


class ContributionExplainer:
    """Wraps any Classifier exposing ``contributions(X) -> (n, p+1)``."""

    def __init__(self, model: Classifier) -> None:
        if not hasattr(model, "contributions"):
            raise TypeError(f"{type(model).__name__} has no contributions()")
        self.model = model

    def explain(self, X: pd.DataFrame) -> Explanation:
        contrib = np.asarray(self.model.contributions(X))
        return Explanation(
            values=contrib[:, :-1],
            base_value=float(contrib[0, -1]) if len(contrib) else 0.0,
            feature_names=list(self.model.feature_names_),
            data=X[self.model.feature_names_],
        )


class ShapLibraryExplainer:
    """Fallback using the ``shap`` library (model-agnostic TreeExplainer)."""

    def __init__(self, model: Classifier) -> None:
        import shap

        self.model = model
        inner = getattr(model, "model_", model)
        self._explainer = shap.TreeExplainer(inner)

    def explain(self, X: pd.DataFrame) -> Explanation:
        Xm = X[self.model.feature_names_]
        values = self._explainer.shap_values(Xm)
        if isinstance(values, list):  # older shap: one array per class
            values = values[1]
        base = self._explainer.expected_value
        base = float(np.atleast_1d(base)[-1])
        return Explanation(
            values=np.asarray(values), base_value=base, feature_names=list(Xm.columns), data=Xm
        )


def make_explainer(model: Classifier, explain_model: str = "lightgbm") -> ContributionExplainer:
    """For ensembles, explain the configured tree base model."""
    target = model
    base_models = getattr(model, "base_models_", None)
    if base_models:
        target = base_models.get(explain_model) or next(
            m for m in base_models.values() if hasattr(m, "contributions")
        )
    return ContributionExplainer(target)


def global_importance(expl: Explanation) -> pd.Series:
    return pd.Series(np.abs(expl.values).mean(axis=0), index=expl.feature_names).sort_values(
        ascending=False
    )


def shap_summary_plot(expl: Explanation, path: Path, max_display: int = 25) -> Path:
    """Beeswarm via the shap library when available, else a mean |SHAP| bar chart."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import shap

        shap.summary_plot(
            expl.values,
            expl.data,
            feature_names=expl.feature_names,
            max_display=max_display,
            show=False,
        )
        plt.gcf().set_size_inches(8, 0.3 * max_display + 1.5)
        plt.tight_layout()
        plt.savefig(path, dpi=110)
        plt.close("all")
        return path
    except ImportError:
        from risklens.evaluation.plots import importance_plot

        return importance_plot(global_importance(expl), path, top=max_display)


def dependence_plot(expl: Explanation, feature: str, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    j = expl.feature_names.index(feature)
    assert expl.data is not None, "explanation has no feature data"
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(expl.data[feature], expl.values[:, j], s=3, alpha=0.4)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel=feature, ylabel="SHAP value (log-odds)", title=f"Dependence: {feature}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
