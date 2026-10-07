"""Features from the main application table.

``application_features`` is THE function used in both training and serving
(training/serving skew prevention): it keeps every input column and only adds
derived ones, so pre-computed history aggregates pass through untouched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.data.schemas import DAYS_EMPLOYED_ANOMALY
from risklens.features.common import safe_div

AGE_BINS = [0, 25, 35, 45, 55, 200]
AGE_LABELS = ["<25", "25-34", "35-44", "45-54", "55+"]


def age_group(days_birth: pd.Series) -> pd.Series:
    years = -pd.to_numeric(days_birth, errors="coerce") / 365.25
    return pd.cut(years, bins=AGE_BINS, labels=AGE_LABELS, right=False).astype("object")


def application_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # --- data repairs
    if "DAYS_EMPLOYED" in out:
        emp = pd.to_numeric(out["DAYS_EMPLOYED"], errors="coerce")
        out["DAYS_EMPLOYED_ANOM"] = (emp == DAYS_EMPLOYED_ANOMALY).astype("int8")
        out["DAYS_EMPLOYED"] = emp.where(emp != DAYS_EMPLOYED_ANOMALY)
    if "CODE_GENDER" in out:
        out["CODE_GENDER"] = out["CODE_GENDER"].where(out["CODE_GENDER"] != "XNA")
    if "ORGANIZATION_TYPE" in out:  # 'XNA' = not applicable (pensioners)
        out["ORGANIZATION_TYPE"] = out["ORGANIZATION_TYPE"].where(out["ORGANIZATION_TYPE"] != "XNA")

    def col(name: str) -> pd.Series:
        if name in out:
            return pd.to_numeric(out[name], errors="coerce")
        return pd.Series(np.nan, index=out.index)

    income, credit = col("AMT_INCOME_TOTAL"), col("AMT_CREDIT")
    annuity, goods = col("AMT_ANNUITY"), col("AMT_GOODS_PRICE")
    birth, employed = col("DAYS_BIRTH"), col("DAYS_EMPLOYED")

    # --- affordability ratios
    out["CREDIT_TO_INCOME"] = safe_div(credit, income)
    out["ANNUITY_TO_INCOME"] = safe_div(annuity, income)
    out["CREDIT_TO_GOODS"] = safe_div(credit, goods)
    out["CREDIT_TERM"] = safe_div(annuity, credit)  # inverse of #payments
    out["GOODS_MINUS_CREDIT"] = goods - credit
    out["INCOME_PER_PERSON"] = safe_div(income, col("CNT_FAM_MEMBERS"))
    out["INCOME_PER_CHILD"] = safe_div(income, 1 + col("CNT_CHILDREN"))

    # --- time features
    out["AGE_YEARS"] = -birth / 365.25
    out["EMPLOYED_TO_AGE"] = safe_div(employed, birth)
    out["REGISTRATION_TO_AGE"] = safe_div(col("DAYS_REGISTRATION"), birth)
    out["ID_PUBLISH_TO_AGE"] = safe_div(col("DAYS_ID_PUBLISH"), birth)
    out["PHONE_CHANGE_TO_AGE"] = safe_div(col("DAYS_LAST_PHONE_CHANGE"), birth)
    out["CAR_TO_AGE"] = safe_div(col("OWN_CAR_AGE"), out["AGE_YEARS"])
    out["AGE_GROUP"] = age_group(birth) if "DAYS_BIRTH" in out else np.nan

    # --- external bureau scores: the strongest signal in this dataset
    ext_cols = [c for c in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3") if c in out]
    if ext_cols:
        ext = out[ext_cols].apply(pd.to_numeric, errors="coerce").astype("float64")
        out["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        out["EXT_SOURCE_STD"] = ext.std(axis=1)
        out["EXT_SOURCE_MIN"] = ext.min(axis=1)
        out["EXT_SOURCE_MAX"] = ext.max(axis=1)
        out["EXT_SOURCE_PROD"] = ext.prod(axis=1, min_count=len(ext_cols))
        out["EXT_SOURCE_NAN_COUNT"] = ext.isna().sum(axis=1)
        weights = np.array([2.0, 1.0, 3.0][: len(ext_cols)])
        values = ext.to_numpy()
        row_mean = out["EXT_SOURCE_MEAN"].to_numpy()[:, None]
        filled = np.where(np.isnan(values), row_mean, values)  # impute with the row's own mean
        out["EXT_SOURCE_WEIGHTED"] = (filled * weights).sum(axis=1) / weights.sum()
        out["CREDIT_TO_ANNUITY_X_EXT"] = safe_div(credit, annuity) * out["EXT_SOURCE_MEAN"]

    # --- document / contact flag counts
    doc_cols = [c for c in out.columns if c.startswith("FLAG_DOCUMENT_")]
    if doc_cols:
        out["DOCUMENT_COUNT"] = out[doc_cols].sum(axis=1)
    contact_cols = [
        c
        for c in (
            "FLAG_MOBIL",
            "FLAG_EMP_PHONE",
            "FLAG_WORK_PHONE",
            "FLAG_CONT_MOBILE",
            "FLAG_PHONE",
            "FLAG_EMAIL",
        )
        if c in out
    ]
    if contact_cols:
        out["CONTACT_COUNT"] = out[contact_cols].sum(axis=1)
    bureau_req = [c for c in out.columns if c.startswith("AMT_REQ_CREDIT_BUREAU_")]
    if bureau_req:
        out["BUREAU_REQ_TOTAL"] = out[bureau_req].sum(axis=1, min_count=1)

    for c in out.columns:
        if out[c].dtype == "float64":
            with np.errstate(over="ignore"):
                v = out[c].to_numpy().astype("float32")
            v[np.isinf(v)] = np.nan  # finite float64 can overflow float32
            out[c] = v
    return out


class ApplicationFeatures:
    name = "application"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return application_features(tables["application"])
