"""Feature selection: missing / constant / collinear filters, then LightGBM
gain importance (optionally permutation importance) -> keep top-k."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from risklens.config import SelectionConfig

log = logging.getLogger(__name__)


class FeatureSelector:
    def __init__(self, cfg: SelectionConfig, seed: int = 42) -> None:
        self.cfg = cfg
        self.seed = seed

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_valid: pd.DataFrame | None = None,
        y_valid: pd.Series | None = None,
    ) -> FeatureSelector:
        rows = []
        cols = list(X.columns)
        if not self.cfg.enabled:
            self.selected_ = cols
            self.report_ = pd.DataFrame({"feature": cols, "kept": True})
            return self

        missing = X.isna().mean()
        nunique = X.nunique(dropna=True)
        dropped: dict[str, str] = {}
        for c in cols:
            if missing[c] > self.cfg.max_missing:
                dropped[c] = f"missing>{self.cfg.max_missing}"
            elif nunique[c] <= 1:
                dropped[c] = "constant"
        remaining = [c for c in cols if c not in dropped]

        for c in self._collinear(X[remaining]):
            dropped[c] = f"corr>{self.cfg.corr_threshold}"
        remaining = [c for c in remaining if c not in dropped]

        gain = self._gain_importance(X[remaining], y, X_valid, y_valid)
        ranked = gain.sort_values(ascending=False)
        top_k = self.cfg.top_k or len(ranked)
        perm = None
        if self.cfg.permutation and X_valid is not None and y_valid is not None:
            candidates = list(ranked.index[: int(top_k * 1.5)])
            perm = self._permutation_importance(X[candidates], y, X_valid[candidates], y_valid)
            ranked = perm.sort_values(ascending=False)
        selected = list(ranked.index[:top_k])
        for c in remaining:
            if c not in selected:
                dropped[c] = "low_importance"

        for c in cols:
            rows.append(
                {
                    "feature": c,
                    "missing_rate": float(missing[c]),
                    "gain": float(gain.get(c, np.nan)),
                    "permutation": float(perm.get(c, np.nan)) if perm is not None else np.nan,
                    "kept": c in selected,
                    "reason": dropped.get(c, ""),
                }
            )
        self.selected_ = selected
        self.report_ = pd.DataFrame(rows).sort_values("gain", ascending=False)
        log.info("feature selection: %d -> %d features", len(cols), len(selected))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.reindex(columns=self.selected_)

    # ----------------------------------------------------------- helpers
    def _collinear(self, X: pd.DataFrame) -> list[str]:
        if X.shape[1] < 2:
            return []
        sample = X.sample(min(len(X), self.cfg.corr_sample), random_state=self.seed)
        filled = sample.fillna(sample.median()).to_numpy(dtype="float64")
        std = filled.std(axis=0)
        filled = filled[:, std > 0]
        names = [c for c, s in zip(X.columns, std, strict=True) if s > 0]
        corr = np.abs(np.corrcoef(filled, rowvar=False))
        upper = np.triu(corr, k=1)
        return [
            names[j] for j in range(len(names)) if (upper[:, j] > self.cfg.corr_threshold).any()
        ]

    def _gain_importance(
        self, X: pd.DataFrame, y: pd.Series, X_valid: pd.DataFrame | None, y_valid: pd.Series | None
    ) -> pd.Series:
        import lightgbm as lgb

        pos = float(np.mean(y))
        model = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=31,
            colsample_bytree=0.5,
            subsample=0.8,
            subsample_freq=1,
            min_child_samples=50,
            scale_pos_weight=(1 - pos) / max(pos, 1e-6),
            random_state=self.seed,
            verbose=-1,
            metric="auc",
        )
        if X_valid is not None and y_valid is not None:
            from risklens.models.lgbm import eval_kwargs

            model.fit(
                X,
                y,
                **eval_kwargs(X_valid[X.columns], y_valid),
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
        else:
            model.set_params(n_estimators=300)
            model.fit(X, y)
        self._model = model
        imp = model.booster_.feature_importance(importance_type="gain")
        return pd.Series(imp, index=X.columns, dtype="float64")

    def _permutation_importance(
        self, X: pd.DataFrame, y: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series
    ) -> pd.Series:
        import lightgbm as lgb

        model = lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            colsample_bytree=0.5,
            random_state=self.seed,
            verbose=-1,
        )
        model.fit(X, y)
        rng = np.random.default_rng(self.seed)
        n = min(len(X_valid), self.cfg.permutation_sample)
        idx = rng.choice(len(X_valid), n, replace=False)
        Xv = X_valid.iloc[idx].copy()
        yv = np.asarray(y_valid)[idx]
        base = roc_auc_score(yv, np.asarray(model.predict_proba(Xv))[:, 1])
        out: dict[str, float] = {}
        for c in Xv.columns:
            saved = Xv[c].to_numpy().copy()
            Xv[c] = rng.permutation(saved)
            out[c] = base - roc_auc_score(yv, np.asarray(model.predict_proba(Xv))[:, 1])
            Xv[c] = saved
        return pd.Series(out)
