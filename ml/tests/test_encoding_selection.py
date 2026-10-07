from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.config import SelectionConfig
from risklens.features.encoding import TabularEncoder
from risklens.features.selection import FeatureSelector


def _frame(n=400, seed=0):
    rng = np.random.default_rng(seed)
    cat = rng.choice(["a", "b", "c", None], n)
    y = pd.Series((rng.random(n) < np.where(cat == "a", 0.6, 0.1)).astype(int))
    X = pd.DataFrame(
        {
            "num": rng.normal(size=n),
            "cat": cat,
            "low": rng.choice(["Y", "N"], n),
            "id": np.arange(n),
        }
    )
    return X, y


def test_target_encoding_is_out_of_fold():
    # a category that appears exactly once: its OOF encoding cannot see its own label
    X = pd.DataFrame({"cat": ["unique"] + ["x"] * 99})
    y = pd.Series([1] + [0] * 99)
    enc = TabularEncoder(target_encode=["cat"], n_folds=5, smoothing=1.0)
    out = enc.fit_transform(X, y)
    assert out.loc[0, "cat_TE"] == np.float32(enc.prior_)  # unseen in other folds -> prior
    # transform (inference) uses the full-train map, which DOES include the label
    assert enc.transform(X.head(1)).loc[0, "cat_TE"] > enc.prior_


def test_encoder_columns_stable_and_numeric():
    X, y = _frame()
    enc = TabularEncoder(target_encode=["cat"], drop=["id"], one_hot_max_cardinality=3)
    tr = enc.fit_transform(X, y)
    te = enc.transform(X.head(5))
    assert list(tr.columns) == list(te.columns)
    assert "id" not in tr.columns
    assert {"low=Y", "low=N", "cat_FREQ", "cat_TE"} <= set(tr.columns)
    assert all(dt == np.float32 for dt in tr.dtypes)


def test_unseen_category_and_missing_columns():
    X, y = _frame()
    enc = TabularEncoder(target_encode=["cat"]).fit(X, y)
    new = pd.DataFrame({"cat": ["zzz"]})  # 'num', 'low' missing entirely
    out = enc.transform(new)
    assert out.loc[0, "cat_TE"] == np.float32(enc.prior_)
    assert out.loc[0, "cat_FREQ"] == 0.0
    assert np.isnan(out.loc[0, "num"])


def test_selector_drops_constant_missing_and_collinear():
    rng = np.random.default_rng(0)
    n = 600
    a = rng.normal(size=n)
    y = pd.Series((a + rng.normal(scale=0.5, size=n) > 0).astype(int))
    X = pd.DataFrame(
        {
            "signal": a,
            "copy": a * 2 + 1e-9,
            "const": 1.0,
            "mostly_nan": np.where(rng.random(n) < 0.99, np.nan, 1.0),
            "noise": rng.normal(size=n),
        }
    )
    sel = FeatureSelector(SelectionConfig(top_k=2), seed=0).fit(X, y)
    assert "signal" in sel.selected_
    assert "const" not in sel.selected_ and "mostly_nan" not in sel.selected_
    assert not {"signal", "copy"} <= set(sel.selected_)
    reasons = sel.report_.set_index("feature")["reason"]
    assert reasons["const"] == "constant"
