"""Monthly credit-card snapshots: utilisation, drawings, DPD trends."""

from __future__ import annotations

import pandas as pd

from risklens.features.common import aggregate, group_slope, safe_div

KEY = "SK_ID_CURR"


def credit_card_features(cc: pd.DataFrame) -> pd.DataFrame:
    if cc.empty:
        return pd.DataFrame(index=pd.Index([], name=KEY))
    c = cc.copy()
    c["UTILIZATION"] = safe_div(c["AMT_BALANCE"], c["AMT_CREDIT_LIMIT_ACTUAL"])
    c["ATM_SHARE"] = safe_div(c.get("AMT_DRAWINGS_ATM_CURRENT"), c.get("AMT_DRAWINGS_CURRENT"))
    c["PAYMENT_TO_MIN"] = safe_div(
        c.get("AMT_PAYMENT_TOTAL_CURRENT"), c.get("AMT_INST_MIN_REGULARITY")
    )
    c["IS_LATE"] = (c["SK_DPD"] > 0).astype("float32")
    spec = {
        "MONTHS_BALANCE": ["min", "size"],
        "AMT_BALANCE": ["mean", "max"],
        "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "max"],
        "UTILIZATION": ["mean", "max"],
        "ATM_SHARE": ["mean"],
        "PAYMENT_TO_MIN": ["mean", "min"],
        "AMT_DRAWINGS_CURRENT": ["mean", "sum"],
        "AMT_TOTAL_RECEIVABLE": ["mean"],
        "CNT_DRAWINGS_CURRENT": ["mean", "max"],
        "SK_DPD": ["mean", "max"],
        "SK_DPD_DEF": ["max"],
        "IS_LATE": ["mean"],
    }
    out = aggregate(c, KEY, spec, "CC")
    out["CC_UTILIZATION_SLOPE"] = group_slope(c, KEY, "MONTHS_BALANCE", "UTILIZATION").astype(
        "float32"
    )
    recent = aggregate(
        c[c["MONTHS_BALANCE"] >= -6], KEY, {"UTILIZATION": ["mean"], "SK_DPD": ["max"]}, "CC_6M"
    )
    return out.join(recent, how="left").astype("float32")


class CreditCardFeatures:
    name = "credit_card"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return credit_card_features(tables["credit_card"])
