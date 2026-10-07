"""Categorical encoding without leakage.

* target encoding with K-fold out-of-fold estimates on the training set
  (a row's own label never influences its own encoding) + smoothing
* frequency encoding for every categorical
* one-hot for low-cardinality categoricals

Output is an all-numeric float32 frame with a fixed column order, so every
model (LogReg, GBDTs, MLP) consumes the same matrix.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

NAN_TOKEN = "__nan__"


def _as_category_str(s: pd.Series) -> pd.Series:
    return s.astype("object").where(s.notna(), NAN_TOKEN).astype(str)


def _category_array(s: pd.Series) -> np.ndarray:
    """Fast numpy equivalent of ``_as_category_str`` (used on the hot path)."""
    arr = s.to_numpy(dtype=object)
    mask = pd.isna(arr)
    out = arr.astype(str)
    out[mask] = NAN_TOKEN
    return out


def _lookup(values: np.ndarray, mapping: dict[str, float], default: float) -> np.ndarray:
    get = mapping.get
    return np.fromiter((get(v, default) for v in values), dtype="float32", count=len(values))


def smoothed_means(cat: pd.Series, y: pd.Series, prior: float, smoothing: float) -> pd.Series:
    stats = (
        pd.DataFrame({"c": cat.to_numpy(), "y": y.to_numpy()})
        .groupby("c")["y"]
        .agg(["sum", "count"])
    )
    return (stats["sum"] + smoothing * prior) / (stats["count"] + smoothing)


class TabularEncoder:
    def __init__(
        self,
        target_encode: Iterable[str] = (),
        one_hot_max_cardinality: int = 10,
        n_folds: int = 5,
        smoothing: float = 20.0,
        seed: int = 42,
        drop: Iterable[str] = (),
    ) -> None:
        self.target_encode = list(target_encode)
        self.one_hot_max_cardinality = one_hot_max_cardinality
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.seed = seed
        self.drop = list(drop)

    # ------------------------------------------------------------------ fit
    def _fit_structure(self, X: pd.DataFrame) -> None:
        self.__dict__.pop("feature_names_out_", None)  # refit -> recompute output names
        cols = [c for c in X.columns if c not in self.drop]
        self.cat_cols_ = [
            c
            for c in cols
            if X[c].dtype == "object"
            or isinstance(X[c].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X[c])
            or pd.api.types.is_bool_dtype(X[c])
        ]
        self.num_cols_ = [c for c in cols if c not in self.cat_cols_]
        self.freq_: dict[str, dict[str, float]] = {}
        self.onehot_: dict[str, list[str]] = {}
        for c in self.cat_cols_:
            s = _as_category_str(X[c])
            vc = s.value_counts(normalize=True)
            self.freq_[c] = vc.astype("float64").to_dict()
            if len(vc) <= self.one_hot_max_cardinality:
                self.onehot_[c] = [v for v in vc.index if v != NAN_TOKEN]
        self.te_cols_ = [c for c in self.target_encode if c in self.cat_cols_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and return the TRAINING encoding (out-of-fold target encoding)."""
        self._fit_structure(X)
        y = pd.Series(np.asarray(y, dtype="float64"), index=X.index)
        self.prior_ = float(y.mean())
        self.te_maps_: dict[str, dict[str, float]] = {}
        oof: dict[str, np.ndarray] = {}
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        for c in self.te_cols_:
            s = _as_category_str(X[c])
            self.te_maps_[c] = smoothed_means(s, y, self.prior_, self.smoothing).to_dict()
            values = np.full(len(X), self.prior_)
            for fit_idx, enc_idx in kf.split(X):
                m = smoothed_means(s.iloc[fit_idx], y.iloc[fit_idx], self.prior_, self.smoothing)
                values[enc_idx] = s.iloc[enc_idx].map(m).fillna(self.prior_).to_numpy()
            oof[c] = values
        return self._transform(X, te_override=oof)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TabularEncoder:
        self.fit_transform(X, y)
        return self

    # ------------------------------------------------------------ transform
    def transform(self, X: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
        """Encode ``X``. With ``columns`` only those outputs are computed (serving
        hot path: the model needs ~250 of the ~450 encoded features)."""
        return self._transform(X, te_override=None, columns=columns)

    def _numeric_block(self, X: pd.DataFrame, cols: list[str]) -> np.ndarray:
        sub = X.reindex(columns=cols)
        bad = [c for c in cols if not pd.api.types.is_numeric_dtype(sub[c])]
        if bad:  # e.g. all-None JSON columns arrive as object dtype
            sub = sub.copy()
            for c in bad:
                sub[c] = pd.to_numeric(sub[c], errors="coerce")
        return sub.to_numpy(dtype="float32")

    def _transform(
        self,
        X: pd.DataFrame,
        te_override: dict[str, np.ndarray] | None,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        wanted = set(columns) if columns is not None else None
        num_cols = [c for c in self.num_cols_ if wanted is None or _clean_name(c) in wanted]
        block = self._numeric_block(X, num_cols)
        cols: dict[str, np.ndarray] = {c: block[:, j] for j, c in enumerate(num_cols)}

        def need(name: str) -> bool:
            return wanted is None or _clean_name(name) in wanted

        for c in self.cat_cols_:
            outputs = [f"{c}_FREQ", *(f"{c}={v}" for v in self.onehot_.get(c, [])), f"{c}_TE"]
            if not any(need(o) for o in outputs):
                continue
            s = _category_array(X[c]) if c in X else np.full(len(X), NAN_TOKEN, dtype=object)
            if need(f"{c}_FREQ"):
                cols[f"{c}_FREQ"] = _lookup(s, self.freq_[c], 0.0)
            for v in self.onehot_.get(c, []):
                if need(f"{c}={v}"):
                    cols[f"{c}={v}"] = (s == v).astype("float32")
            if c in self.te_cols_ and need(f"{c}_TE"):
                if te_override is not None:
                    cols[f"{c}_TE"] = te_override[c].astype("float32")
                else:
                    cols[f"{c}_TE"] = _lookup(s, self.te_maps_[c], self.prior_)
        out = pd.DataFrame({_clean_name(k): v for k, v in cols.items()}, index=X.index)
        if not hasattr(self, "feature_names_out_"):
            self.feature_names_out_ = list(out.columns)
        return out[columns if columns is not None else self.feature_names_out_]

    def get_feature_names_out(self) -> list[str]:
        return list(self.feature_names_out_)


def _clean_name(name: str) -> str:
    """LightGBM rejects JSON special characters in feature names."""
    for ch in '[]{}":,':
        name = name.replace(ch, "_")
    return name.replace(" ", "_")
