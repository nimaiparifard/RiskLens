"""Monthly POS / cash loan snapshots: DPD levels and trends."""

from __future__ import annotations

import pandas as pd

from risklens.features.common import aggregate, group_slope

KEY = "SK_ID_CURR"


def pos_cash_features(pos: pd.DataFrame) -> pd.DataFrame:
    if pos.empty:
        return pd.DataFrame(index=pd.Index([], name=KEY))
    p = pos.copy()
    p["IS_LATE"] = (p["SK_DPD"] > 0).astype("float32")
    p["IS_COMPLETED"] = (p["NAME_CONTRACT_STATUS"] == "Completed").astype("float32")
    p["REMAINING_SHARE"] = p["CNT_INSTALMENT_FUTURE"] / p["CNT_INSTALMENT"].where(
        p["CNT_INSTALMENT"] > 0
    )
    spec = {
        "MONTHS_BALANCE": ["min", "max", "size"],
        "SK_DPD": ["mean", "max"],
        "SK_DPD_DEF": ["mean", "max"],
        "IS_LATE": ["mean"],
        "IS_COMPLETED": ["mean"],
        "CNT_INSTALMENT": ["mean"],
        "CNT_INSTALMENT_FUTURE": ["mean", "sum"],
        "REMAINING_SHARE": ["mean"],
    }
    out = aggregate(p, KEY, spec, "POS")
    out["POS_LOANS_COUNT"] = p.groupby(KEY)["SK_ID_PREV"].nunique().astype("float32")
    out["POS_DPD_SLOPE"] = group_slope(p, KEY, "MONTHS_BALANCE", "SK_DPD").astype("float32")
    recent = aggregate(
        p[p["MONTHS_BALANCE"] >= -12],
        KEY,
        {"SK_DPD": ["mean", "max"], "IS_LATE": ["mean"]},
        "POS_12M",
    )
    return out.join(recent, how="left").astype("float32")


class PosCashFeatures:
    name = "pos_cash"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pos_cash_features(tables["pos_cash"])
