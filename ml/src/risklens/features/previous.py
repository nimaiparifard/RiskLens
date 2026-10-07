"""Previous Home Credit applications: approval/refusal history, requested vs granted."""

from __future__ import annotations

import pandas as pd

from risklens.features.common import aggregate, rate_by_category, replace_sentinel, safe_div

KEY = "SK_ID_CURR"
DAY_COLS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]
STATUSES = ["Approved", "Refused", "Canceled", "Unused offer"]


def previous_features(prev: pd.DataFrame) -> pd.DataFrame:
    if prev.empty:
        return pd.DataFrame(index=pd.Index([], name=KEY))
    p = replace_sentinel(prev, DAY_COLS)
    p["APP_TO_CREDIT"] = safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])  # requested / granted
    p["CREDIT_TO_ANNUITY"] = safe_div(p["AMT_CREDIT"], p["AMT_ANNUITY"])
    p["DOWN_PAYMENT_RATIO"] = safe_div(p.get("AMT_DOWN_PAYMENT"), p["AMT_CREDIT"])
    p["GOODS_TO_CREDIT"] = safe_div(p.get("AMT_GOODS_PRICE"), p["AMT_CREDIT"])
    if "DAYS_LAST_DUE_1ST_VERSION" in p and "DAYS_LAST_DUE" in p:
        p["LAST_DUE_DIFF"] = p["DAYS_LAST_DUE_1ST_VERSION"] - p["DAYS_LAST_DUE"]

    spec = {
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max", "sum"],
        "AMT_CREDIT": ["mean", "max", "sum"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "APP_TO_CREDIT": ["mean", "min", "max"],
        "CREDIT_TO_ANNUITY": ["mean"],
        "DOWN_PAYMENT_RATIO": ["mean"],
        "GOODS_TO_CREDIT": ["mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
        "DAYS_TERMINATION": ["max"],
        "LAST_DUE_DIFF": ["mean"],
    }
    out = aggregate(p, KEY, spec, "PREV")
    out["PREV_COUNT"] = p.groupby(KEY).size().astype("float32")
    rates = rate_by_category(p, KEY, "NAME_CONTRACT_STATUS", "PREV", STATUSES)
    rates.columns = [
        c.replace("NAME_CONTRACT_STATUS_", "").replace("_SHARE", "_RATE") for c in rates.columns
    ]  # PREV_APPROVED_RATE, PREV_REFUSED_RATE, ...
    yields = rate_by_category(p, KEY, "NAME_YIELD_GROUP", "PREV", ["high", "low_normal", "middle"])

    approved = aggregate(
        p[p["NAME_CONTRACT_STATUS"] == "Approved"],
        KEY,
        {
            "AMT_CREDIT": ["mean", "sum"],
            "APP_TO_CREDIT": ["mean"],
            "DAYS_DECISION": ["max"],
            "CNT_PAYMENT": ["mean"],
        },
        "APPROVED",
    )
    refused = aggregate(
        p[p["NAME_CONTRACT_STATUS"] == "Refused"],
        KEY,
        {"AMT_APPLICATION": ["mean", "max"], "DAYS_DECISION": ["max", "mean"]},
        "REFUSED",
    )
    recent = p[p["DAYS_DECISION"] >= -365]
    last12 = pd.DataFrame(
        {
            "PREV_12M_COUNT": recent.groupby(KEY).size(),
            "PREV_12M_REFUSED_COUNT": recent[recent["NAME_CONTRACT_STATUS"] == "Refused"]
            .groupby(KEY)
            .size(),
        }
    ).astype("float32")
    return out.join([rates, yields, approved, refused, last12], how="left").astype("float32")


class PreviousFeatures:
    name = "previous"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return previous_features(tables["previous"])
