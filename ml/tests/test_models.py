from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from risklens.models.base import apply_imbalance, focal_loss_objective, positive_weight
from risklens.models.calibration import Calibrator
from risklens.registry import available_models, create_model


@pytest.fixture(scope="module")
def xy():
    rng = np.random.default_rng(0)
    n = 1500
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    X.loc[rng.random(n) < 0.1, "f2"] = np.nan
    logit = -2 + 1.5 * X["f0"] - X["f1"] + 0.5 * X["f2"].fillna(0)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X.iloc[:1000], y.iloc[:1000], X.iloc[1000:], y.iloc[1000:]


FAST = {
    "logreg_woe": {"max_bins": 5},
    "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "early_stopping_rounds": 20},
    "xgboost": {"n_estimators": 100, "learning_rate": 0.1, "early_stopping_rounds": 20},
    "catboost": {"iterations": 100, "early_stopping_rounds": 20},
}


@pytest.mark.parametrize("name", list(FAST))
def test_model_learns_and_explains(name, xy):
    if name == "catboost":
        pytest.importorskip("catboost")
    X, y, Xv, yv = xy
    m = create_model(name, **FAST[name], pos_weight=positive_weight(y)).fit(X, y, Xv, yv)
    p = m.predict_proba(Xv)
    assert p.shape == (len(Xv),) and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(yv, p) > 0.75
    contrib = m.contributions(Xv.head(3))
    assert contrib.shape == (3, X.shape[1] + 1)
    imp = m.feature_importance()
    assert imp is not None and imp.index[0] in ("f0", "f1")


def test_stacking(xy):
    X, y, Xv, yv = xy
    m = create_model(
        "stacking",
        base_models=["logreg_woe", "lightgbm"],
        n_folds=3,
        base_params={"logreg_woe": FAST["logreg_woe"], "lightgbm": FAST["lightgbm"]},
    ).fit(X, y, Xv, yv)
    assert roc_auc_score(yv, m.predict_proba(Xv)) > 0.75
    assert set(m.meta_weights()) == {"logreg_woe", "lightgbm"}
    assert not np.isnan(m.oof_).any()


def test_stacking_reuses_prefit(xy):
    X, y, Xv, yv = xy
    lgbm = create_model("lightgbm", **FAST["lightgbm"]).fit(X, y, Xv, yv)
    m = create_model(
        "stacking", base_models=["lightgbm"], n_folds=2, base_params={"lightgbm": FAST["lightgbm"]}
    )
    m.set_prefit({"lightgbm": lgbm}).fit(X, y, Xv, yv)
    assert m.base_models_["lightgbm"] is lgbm


def test_woe_scorecard(xy):
    X, y, _Xv, _yv = xy
    m = create_model("logreg_woe", max_bins=5).fit(X, y)
    card = m.scorecard()
    assert {"feature", "bin", "woe", "points"} <= set(card.columns)
    assert m.binner_.iv_["f0"] > m.binner_.iv_["f5"]  # informative > noise


def test_focal_loss_lightgbm(xy):
    X, y, Xv, yv = xy
    m = create_model("lightgbm", n_estimators=80, learning_rate=0.1, focal_gamma=2.0).fit(X, y)
    assert roc_auc_score(yv, m.predict_proba(Xv)) > 0.7


def test_focal_gradient_sign():
    obj = focal_loss_objective(2.0, 0.5)
    grad, hess = obj(np.array([1.0, 0.0]), np.array([0.0, 0.0]))
    assert grad[0] < 0 < grad[1]  # push positive up, negative down
    assert (hess > 0).all()


@pytest.mark.parametrize("strategy", ["none", "scale_pos_weight", "undersample", "smote"])
def test_imbalance_strategies(strategy, xy):
    X, y, *_ = xy
    if strategy == "smote":
        pytest.importorskip("imblearn")
    _X2, y2, extra = apply_imbalance(strategy, X, y, 0)
    if strategy == "scale_pos_weight":
        assert extra["pos_weight"] > 1
    if strategy == "undersample":
        assert y2.mean() > y.mean()
    if strategy == "smote":
        assert len(y2) > len(y)


@pytest.mark.parametrize("method", ["isotonic", "sigmoid"])
def test_calibration_improves_brier(method, xy):
    from sklearn.metrics import brier_score_loss

    X, y, Xv, yv = xy
    m = create_model("lightgbm", **FAST["lightgbm"], pos_weight=10.0).fit(X, y, Xv, yv)
    p = m.predict_proba(Xv)
    cal = Calibrator(method).fit(p, yv)
    assert brier_score_loss(yv, cal.transform(p)) < brier_score_loss(yv, p)


def test_registry_lists_models():
    assert {"lightgbm", "xgboost", "catboost", "logreg_woe", "stacking"} <= set(available_models())
    with pytest.raises(KeyError):
        create_model("nope")
