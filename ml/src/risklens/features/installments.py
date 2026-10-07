"""Repayment history: lateness, days past due, payment ratios."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.features.common import aggregate, safe_div

KEY = "SK_ID_CURR"


def installment_rows(inst: pd.DataFrame) -> pd.DataFrame:
    """Row-level derived columns (exposed for unit tests)."""
    i = inst.copy()
    delay = i["DAYS_ENTRY_PAYMENT"] - i["DAYS_INSTALMENT"]
    i["DPD"] = np.clip(delay, 0, None)  # days past due
    i["DBD"] = np.clip(-delay, 0, None)  # days before due
    i["IS_LATE"] = (delay > 0).astype("float32")
    i["IS_LATE_30"] = (delay > 30).astype("float32")
    i["PAYMENT_RATIO"] = safe_div(i["AMT_PAYMENT"], i["AMT_INSTALMENT"])
    i["PAYMENT_DIFF"] = i["AMT_INSTALMENT"] - i["AMT_PAYMENT"]
    i["IS_UNDERPAID"] = (i["PAYMENT_DIFF"] > 1).astype("float32")
    return i


def installments_features(inst: pd.DataFrame) -> pd.DataFrame:
    if inst.empty:
        return pd.DataFrame(index=pd.Index([], name=KEY))
    i = installment_rows(inst)
    spec = {
        "DPD": ["mean", "max", "sum"],
        "DBD": ["mean", "max"],
        "IS_LATE": ["mean", "sum"],
        "IS_LATE_30": ["mean"],
        "IS_UNDERPAID": ["mean"],
        "PAYMENT_RATIO": ["mean", "min"],
        "PAYMENT_DIFF": ["mean", "max", "sum"],
        "AMT_INSTALMENT": ["mean", "max", "sum"],
        "AMT_PAYMENT": ["mean", "sum"],
        "DAYS_ENTRY_PAYMENT": ["max", "mean"],
        "NUM_INSTALMENT_VERSION": ["nunique"],
    }
    out = aggregate(i, KEY, spec, "INSTAL")
    out["INSTAL_COUNT"] = i.groupby(KEY).size().astype("float32")
    for days in (365, 730):
        w = i[i["DAYS_INSTALMENT"] >= -days]
        agg = aggregate(
            w,
            KEY,
            {"IS_LATE": ["mean"], "DPD": ["mean", "max"], "PAYMENT_RATIO": ["mean"]},
            f"INSTAL_{days // 30}M",
        )
        out = out.join(agg, how="left")
    out["INSTAL_LATE_TREND"] = out.get("INSTAL_12M_IS_LATE_MEAN") - out.get("INSTAL_IS_LATE_MEAN")
    return out.astype("float32")


class InstallmentsFeatures:
    name = "installments"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return installments_features(tables["installments"])
