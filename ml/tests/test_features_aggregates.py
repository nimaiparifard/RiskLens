from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.features.bureau import bureau_balance_per_credit, bureau_features
from risklens.features.common import aggregate, group_slope
from risklens.features.credit_card import credit_card_features
from risklens.features.installments import installment_rows, installments_features
from risklens.features.pos_cash import pos_cash_features
from risklens.features.previous import previous_features
from risklens.features.registry import build_credit_features


def test_group_slope_exact():
    df = pd.DataFrame({"k": [1, 1, 1, 2, 2, 3], "x": [0, 1, 2, 0, 1, 5], "y": [1, 3, 5, 2, 2, 7]})
    s = group_slope(df, "k", "x", "y")
    assert s[1] == 2.0
    assert s[2] == 0.0
    assert np.isnan(s[3])  # single point


def test_aggregate_skips_missing_columns():
    df = pd.DataFrame({"k": [1, 1, 2], "a": [1.0, 3.0, 5.0]})
    out = aggregate(df, "k", {"a": ["mean", "max"], "missing": ["sum"]}, "P")
    assert list(out.columns) == ["P_A_MEAN", "P_A_MAX"]
    assert out.loc[1, "P_A_MEAN"] == 2.0


def test_installments_dpd_and_ratios():
    inst = pd.DataFrame(
        {
            "SK_ID_PREV": [1, 1, 2],
            "SK_ID_CURR": [10, 10, 10],
            "NUM_INSTALMENT_VERSION": [1, 1, 1],
            "NUM_INSTALMENT_NUMBER": [1, 2, 1],
            "DAYS_INSTALMENT": [-100.0, -70.0, -40.0],
            "DAYS_ENTRY_PAYMENT": [-105.0, -60.0, -40.0],
            "AMT_INSTALMENT": [100.0, 100.0, 50.0],
            "AMT_PAYMENT": [100.0, 50.0, 50.0],
        }
    )
    rows = installment_rows(inst)
    assert rows["DPD"].tolist() == [0.0, 10.0, 0.0]
    assert rows["DBD"].tolist() == [5.0, 0.0, 0.0]
    assert rows["PAYMENT_RATIO"].tolist() == [1.0, 0.5, 1.0]
    agg = installments_features(inst)
    assert np.isclose(agg.loc[10, "INSTAL_IS_LATE_MEAN"], 1 / 3)
    assert agg.loc[10, "INSTAL_DPD_MAX"] == 10.0
    assert agg.loc[10, "INSTAL_COUNT"] == 3


def test_bureau_balance_and_bureau(credit_tables):
    bb = bureau_balance_per_credit(credit_tables["bureau_balance"])
    assert bb["BB_LATE_SHARE"].between(0, 1).all()
    feats = bureau_features(credit_tables["bureau"], credit_tables["bureau_balance"])
    assert feats.index.name == "SK_ID_CURR"
    assert {
        "BURO_COUNT",
        "ACTIVE_AMT_CREDIT_SUM_DEBT_SUM",
        "BURO_12M_COUNT",
        "BURO_BB_LATE_SHARE_MEAN",
    } <= set(feats.columns)
    counts = credit_tables["bureau"].groupby("SK_ID_CURR").size()
    assert (feats["BURO_COUNT"] == counts.reindex(feats.index)).all()


def test_previous_rates_sum_to_one(credit_tables):
    feats = previous_features(credit_tables["previous"])
    rate_cols = [
        c for c in feats.columns if c.endswith("_RATE") and c.startswith("PREV_") and "12M" not in c
    ]
    np.testing.assert_allclose(feats[rate_cols].sum(axis=1), 1.0, rtol=1e-5)


def test_previous_sentinel_removed():
    prev = pd.DataFrame(
        {
            "SK_ID_PREV": [1],
            "SK_ID_CURR": [5],
            "AMT_ANNUITY": [10.0],
            "AMT_APPLICATION": [100.0],
            "AMT_CREDIT": [80.0],
            "NAME_CONTRACT_STATUS": ["Approved"],
            "DAYS_DECISION": [-10],
            "CNT_PAYMENT": [12.0],
            "DAYS_TERMINATION": [365243.0],
        }
    )
    feats = previous_features(prev)
    assert np.isnan(feats.loc[5, "PREV_DAYS_TERMINATION_MAX"])
    assert feats.loc[5, "PREV_APP_TO_CREDIT_MEAN"] == 1.25


def test_pos_and_credit_card(credit_tables):
    pos = pos_cash_features(credit_tables["pos_cash"])
    assert "POS_DPD_SLOPE" in pos
    cc = credit_card_features(credit_tables["credit_card"])
    assert (cc["CC_UTILIZATION_MEAN"].dropna() >= 0).all()


def test_build_credit_features_one_row_per_application(credit_tables):
    sources = ["application", "bureau", "previous", "installments", "pos_cash", "credit_card"]
    df = build_credit_features(credit_tables, sources)
    assert len(df) == len(credit_tables["application"])
    assert df["SK_ID_CURR"].is_unique
    assert not np.isinf(df.select_dtypes("number").to_numpy(dtype="float64")).any()


def test_empty_tables_do_not_crash():
    empty = pd.DataFrame(columns=["SK_ID_CURR", "SK_ID_BUREAU", "CREDIT_ACTIVE"])
    assert bureau_features(empty).empty
