"""Credits at other institutions (bureau.csv + bureau_balance.csv)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.features.common import aggregate, safe_div

KEY = "SK_ID_CURR"

_DPD_STATUS = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5}


def bureau_balance_per_credit(bb: pd.DataFrame) -> pd.DataFrame:
    """Monthly status history -> one row per SK_ID_BUREAU."""
    if bb.empty:
        return pd.DataFrame(index=pd.Index([], name="SK_ID_BUREAU"))
    status = bb["STATUS"].astype(str)
    d = pd.DataFrame(
        {
            "SK_ID_BUREAU": bb["SK_ID_BUREAU"].to_numpy(),
            "MONTHS_BALANCE": bb["MONTHS_BALANCE"].to_numpy(),
            "DPD_LEVEL": status.map(_DPD_STATUS).astype("float32").to_numpy(),
            "IS_LATE": status.isin(["1", "2", "3", "4", "5"]).astype("float32").to_numpy(),
            "IS_CLOSED": (status == "C").astype("float32").to_numpy(),
            "IS_UNKNOWN": (status == "X").astype("float32").to_numpy(),
        }
    )
    g = d.groupby("SK_ID_BUREAU").agg(
        BB_MONTHS=("MONTHS_BALANCE", "size"),
        BB_MONTHS_MIN=("MONTHS_BALANCE", "min"),
        BB_LATE_SHARE=("IS_LATE", "mean"),
        BB_DPD_MAX=("DPD_LEVEL", "max"),
        BB_CLOSED_SHARE=("IS_CLOSED", "mean"),
        BB_UNKNOWN_SHARE=("IS_UNKNOWN", "mean"),
    )
    recent = d[d["MONTHS_BALANCE"] >= -12].groupby("SK_ID_BUREAU")["IS_LATE"].mean()
    g["BB_LATE_SHARE_12M"] = recent
    return g.astype("float32")


def bureau_features(
    bureau: pd.DataFrame, bureau_balance: pd.DataFrame | None = None
) -> pd.DataFrame:
    if bureau.empty:
        return pd.DataFrame(index=pd.Index([], name=KEY))
    b = bureau.copy()
    if bureau_balance is not None and not bureau_balance.empty:
        b = b.merge(
            bureau_balance_per_credit(bureau_balance),
            left_on="SK_ID_BUREAU",
            right_index=True,
            how="left",
        )

    b["DEBT_RATIO"] = safe_div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])
    b["OVERDUE_RATIO"] = safe_div(b["AMT_CREDIT_SUM_OVERDUE"], b["AMT_CREDIT_SUM_DEBT"])
    b["LIMIT_RATIO"] = safe_div(b.get("AMT_CREDIT_SUM_LIMIT", np.nan), b["AMT_CREDIT_SUM"])
    b["CREDIT_DURATION"] = b["DAYS_CREDIT_ENDDATE"] - b["DAYS_CREDIT"]
    b["ENDDATE_DIFF"] = b["DAYS_CREDIT_ENDDATE"] - b.get("DAYS_ENDDATE_FACT", np.nan)
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype("float32")
    b["IS_BAD_DEBT"] = b["CREDIT_ACTIVE"].isin(["Bad debt", "Sold"]).astype("float32")
    b["HAS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype("float32")
    if "CREDIT_TYPE" in b:
        b["IS_CREDIT_CARD"] = (b["CREDIT_TYPE"] == "Credit card").astype("float32")

    num_spec = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["max", "sum"],
        "AMT_CREDIT_SUM_LIMIT": ["mean", "sum"],
        "AMT_ANNUITY": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "DEBT_RATIO": ["mean", "max"],
        "OVERDUE_RATIO": ["mean", "max"],
        "CREDIT_DURATION": ["mean", "max"],
        "ENDDATE_DIFF": ["mean"],
        "IS_ACTIVE": ["mean", "sum"],
        "IS_BAD_DEBT": ["sum"],
        "HAS_OVERDUE": ["mean", "sum"],
        "IS_CREDIT_CARD": ["mean"],
        "BB_MONTHS": ["mean", "sum"],
        "BB_LATE_SHARE": ["mean", "max"],
        "BB_LATE_SHARE_12M": ["mean", "max"],
        "BB_DPD_MAX": ["max", "mean"],
        "BB_CLOSED_SHARE": ["mean"],
        "BB_UNKNOWN_SHARE": ["mean"],
    }
    out = aggregate(b, KEY, num_spec, "BURO")
    out["BURO_COUNT"] = b.groupby(KEY).size().astype("float32")

    active = aggregate(
        b[b["CREDIT_ACTIVE"] == "Active"],
        KEY,
        {
            "DAYS_CREDIT": ["max", "mean"],
            "DAYS_CREDIT_ENDDATE": ["mean"],
            "AMT_CREDIT_SUM": ["sum", "mean"],
            "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
            "AMT_CREDIT_SUM_OVERDUE": ["sum"],
            "DEBT_RATIO": ["mean"],
            "CREDIT_DAY_OVERDUE": ["max"],
        },
        "ACTIVE",
    )
    closed = aggregate(
        b[b["CREDIT_ACTIVE"] == "Closed"],
        KEY,
        {
            "DAYS_CREDIT": ["max", "mean"],
            "AMT_CREDIT_SUM": ["sum", "mean"],
            "ENDDATE_DIFF": ["mean"],
        },
        "CLOSED",
    )
    # time windows: credits opened in the last 12 / 24 months
    windows = []
    for months in (12, 24):
        w = b[b["DAYS_CREDIT"] >= -30.4 * months]
        agg = aggregate(
            w,
            KEY,
            {"AMT_CREDIT_SUM": ["sum"], "AMT_CREDIT_SUM_DEBT": ["sum"], "HAS_OVERDUE": ["sum"]},
            f"BURO_{months}M",
        )
        agg[f"BURO_{months}M_COUNT"] = w.groupby(KEY).size().astype("float32")
        windows.append(agg)

    out = out.join([active, closed, *windows], how="left")
    out["BURO_ACTIVE_DEBT_TO_CREDIT"] = safe_div(
        out.get("ACTIVE_AMT_CREDIT_SUM_DEBT_SUM", np.nan),
        out.get("ACTIVE_AMT_CREDIT_SUM_SUM", np.nan),
    )
    return out.astype("float32")


class BureauFeatures:
    name = "bureau"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return bureau_features(tables["bureau"], tables.get("bureau_balance"))
