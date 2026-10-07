from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames

from risklens.features.application import application_features
from risklens.features.common import safe_div
from risklens.features.registry import prepare_credit_input


def test_days_employed_anomaly_flag_and_nan(tiny_application):
    out = application_features(tiny_application)
    assert out["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1, 0]
    assert np.isnan(out.loc[1, "DAYS_EMPLOYED"])
    assert out.loc[0, "DAYS_EMPLOYED"] == -1000


def test_gender_xna_becomes_missing(tiny_application):
    out = application_features(tiny_application)
    assert pd.isna(out.loc[1, "CODE_GENDER"])


def test_ratios(tiny_application):
    out = application_features(tiny_application)
    assert out.loc[0, "CREDIT_TO_INCOME"] == 3.0
    assert np.isclose(out.loc[0, "ANNUITY_TO_INCOME"], 0.15)
    # zero income / zero goods -> NaN, never inf
    assert np.isnan(out.loc[1, "CREDIT_TO_INCOME"])
    assert np.isnan(out.loc[1, "CREDIT_TO_GOODS"])
    assert np.isnan(out.loc[2, "INCOME_PER_PERSON"])


def test_ext_source_aggregates(tiny_application):
    out = application_features(tiny_application)
    assert np.isclose(out.loc[0, "EXT_SOURCE_MEAN"], 0.55)
    assert out.loc[0, "EXT_SOURCE_NAN_COUNT"] == 1
    assert np.isnan(out.loc[0, "EXT_SOURCE_PROD"])  # product needs all three
    assert out.loc[1, "EXT_SOURCE_MIN"] == out.loc[1, "EXT_SOURCE_MAX"]


def test_document_count_and_age(tiny_application):
    out = application_features(tiny_application)
    assert out["DOCUMENT_COUNT"].tolist() == [2, 0, 1]
    assert out.loc[0, "AGE_GROUP"] == "25-34"
    assert np.isclose(out.loc[1, "AGE_YEARS"], 20_000 / 365.25, atol=1e-3)


def test_input_columns_are_preserved(tiny_application):
    df = tiny_application.assign(BURO_COUNT=[1.0, 2.0, 3.0])
    out = application_features(df)
    assert out["BURO_COUNT"].tolist() == [1.0, 2.0, 3.0]
    assert out["SK_ID_CURR"].tolist() == [1, 2, 3]


def test_does_not_mutate_input(tiny_application):
    before = tiny_application.copy()
    application_features(tiny_application)
    pd.testing.assert_frame_equal(before, tiny_application)


def test_prepare_works_on_partial_payload():
    out = prepare_credit_input(pd.DataFrame([{"AMT_INCOME_TOTAL": 1000.0, "AMT_CREDIT": 2000.0}]))
    assert out.loc[0, "CREDIT_TO_INCOME"] == 2.0


def nan_or(lo: float, hi: float):
    return st.one_of(st.floats(lo, hi), st.just(float("nan")))


@settings(max_examples=150, deadline=None)
@given(
    data_frames(
        [
            column("AMT_INCOME_TOTAL", dtype=float, elements=nan_or(-1e9, 1e9)),
            column("AMT_CREDIT", dtype=float, elements=nan_or(-1e9, 1e9)),
            column("AMT_ANNUITY", dtype=float, elements=nan_or(-1e9, 1e9)),
            column("AMT_GOODS_PRICE", dtype=float, elements=nan_or(-1e9, 1e9)),
            column("DAYS_BIRTH", dtype=float, elements=nan_or(-40000, 0)),
            column("DAYS_EMPLOYED", dtype=float, elements=nan_or(-40000, 400000)),
            column("CNT_FAM_MEMBERS", dtype=float, elements=nan_or(0, 20)),
        ]
    )
)
def test_ratios_never_infinite(df):
    out = application_features(df)
    numeric = out.select_dtypes(include=[np.number]).to_numpy(dtype="float64")
    assert not np.isinf(numeric).any()


@given(st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=1, max_size=30))
def test_safe_div_never_inf(values):
    a = np.array(values)
    b = np.roll(a, 1)
    assert not np.isinf(safe_div(a, b)).any()
