"""IEEE-CIS fraud features.

Every history feature only looks at the PAST of the same card / user id
(expanding windows, shifted by one), so the exact same function can run:
  * offline on the full training table, and
  * online in serving on [history of this card] + [new transaction]
-> no training/serving skew and no look-ahead leakage.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

TIME = "TransactionDT"
AMOUNT = "TransactionAmt"
WINDOWS = {"1H": 3600, "24H": 86400, "7D": 7 * 86400}
_BIG = 10**10  # > max TransactionDT, used to vectorise per-group windows

EMAIL_VENDORS = {
    "gmail": "google",
    "googlemail": "google",
    "yahoo": "yahoo",
    "ymail": "yahoo",
    "rocketmail": "yahoo",
    "hotmail": "microsoft",
    "outlook": "microsoft",
    "live": "microsoft",
    "msn": "microsoft",
    "icloud": "apple",
    "me": "apple",
    "mac": "apple",
    "aol": "aol",
    "aim": "aol",
    "anonymous": "anonymous",
    "protonmail": "protonmail",
}


def uid_columns(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Reconstruct a client id: card1 + addr1 + D1n (day the card was first used)."""
    day = pd.to_numeric(df[TIME]).to_numpy() // 86400
    d1 = (
        pd.to_numeric(df["D1"], errors="coerce").to_numpy(dtype="float64")
        if "D1" in df
        else np.full(len(df), np.nan)
    )
    d1n = (day - d1).astype("float32")
    addr = (
        pd.to_numeric(df["addr1"], errors="coerce").to_numpy(dtype="float64")
        if "addr1" in df
        else np.full(len(df), np.nan)
    )
    card = pd.to_numeric(df["card1"], errors="coerce").to_numpy(dtype="float64")
    uid = np.array(
        [f"{c:.0f}_{a}_{d}" for c, a, d in zip(card, addr, d1n, strict=True)], dtype=object
    )
    return {"D1n": d1n, "uid": uid}


def add_uid(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**uid_columns(df))


def entity_history_features(df: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    """Expanding / sliding-window features over PRIOR transactions of ``key``.

    Fully vectorised: sort by (key, time), then use cumulative sums and
    ``searchsorted`` on a composite (group, time) key for time windows.
    """
    n = len(df)
    codes = pd.factorize(df[key].astype(str))[0].astype(np.int64)
    t = df[TIME].to_numpy(dtype=np.int64)
    a = df[AMOUNT].to_numpy(dtype=np.float64)
    order = np.lexsort((np.arange(n), t, codes))  # stable: group, time, original position
    g, ts, amt = codes[order], t[order], a[order]

    pos = np.arange(n)
    new_group = np.r_[True, g[1:] != g[:-1]]
    group_start = np.maximum.accumulate(np.where(new_group, pos, 0))
    count = (pos - group_start).astype(np.float64)  # number of prior transactions

    csum = np.r_[0.0, np.cumsum(amt)]
    csum2 = np.r_[0.0, np.cumsum(amt * amt)]
    prior_sum = csum[pos] - csum[group_start]
    prior_sum2 = csum2[pos] - csum2[group_start]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, prior_sum / count, np.nan)
        var = np.where(count > 1, (prior_sum2 - count * mean**2) / (count - 1), np.nan)
    std = np.sqrt(np.clip(var, 0, None))

    since_last = np.where(new_group, np.nan, ts - np.r_[ts[0], ts[:-1]]).astype(np.float64)

    composite = g * _BIG + ts
    with np.errstate(divide="ignore", invalid="ignore"):  # masked by np.where
        feats: dict[str, np.ndarray] = {
            f"{prefix}_TX_COUNT": count,
            f"{prefix}_AMT_MEAN": mean,
            f"{prefix}_AMT_STD": std,
            f"{prefix}_AMT_TO_MEAN": np.where(mean > 0, amt / mean, np.nan),
            f"{prefix}_AMT_ZSCORE": np.where(std > 0, (amt - mean) / std, np.nan),
            f"{prefix}_SECS_SINCE_LAST": since_last,
        }
    for name, w in WINDOWS.items():
        start = np.searchsorted(composite, composite - w, side="left")
        start = np.maximum(start, group_start)
        feats[f"{prefix}_COUNT_{name}"] = (pos - start).astype(np.float64)
        feats[f"{prefix}_AMT_SUM_{name}"] = csum[pos] - csum[start]

    inverse = np.empty(n, dtype=np.int64)
    inverse[order] = pos
    return pd.DataFrame(
        {name: values[inverse].astype("float32") for name, values in feats.items()}, index=df.index
    )


def email_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in ("P_emaildomain", "R_emaildomain"):
        if col not in df:
            continue
        dom = df[col].astype("object")
        out[f"{col}_vendor"] = dom.map(
            lambda d: (
                EMAIL_VENDORS.get(str(d).split(".")[0], "other") if isinstance(d, str) else np.nan
            )
        )
        out[f"{col}_suffix"] = dom.map(
            lambda d: d.split(".")[-1] if isinstance(d, str) and "." in d else np.nan
        )
    if "P_emaildomain" in df and "R_emaildomain" in df:
        out["EMAIL_MATCH"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("float32")
        out.loc[df["R_emaildomain"].isna(), "EMAIL_MATCH"] = np.nan
    return out


_DEVICE_PATTERNS = [
    (r"^sm-|samsung|gt-", "samsung"),
    (r"^moto|xt\d", "motorola"),
    (r"^lg", "lg"),
    (r"huawei|^ale-|^-l", "huawei"),
    (r"windows|trident", "windows"),
    (r"ios", "ios"),
    (r"macos", "macos"),
    (r"^rv:", "firefox"),
    (r"pixel|nexus", "google"),
    (r"redmi|^mi ", "xiaomi"),
    (r"linux", "linux"),
    (r"htc", "htc"),
    (r"blade|zte", "zte"),
]


def device_family(info: object) -> object:
    if not isinstance(info, str):
        return np.nan
    s = info.lower()
    for pat, fam in _DEVICE_PATTERNS:
        if re.search(pat, s):
            return fam
    return "other"


def fraud_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw transactions (any order) -> model features (same row order)."""
    t = pd.to_numeric(df[TIME]).to_numpy()
    amt = pd.to_numeric(df[AMOUNT], errors="coerce").to_numpy(dtype="float64")
    extra: dict[str, np.ndarray] = {
        **uid_columns(df),
        "HOUR": ((t // 3600) % 24).astype("float32"),
        "DOW": ((t // 86400) % 7).astype("float32"),
        "AMT_LOG": np.log1p(amt).astype("float32"),
        "AMT_CENTS": np.round(amt - np.floor(amt), 3).astype("float32"),
    }
    if "DeviceInfo" in df:
        extra["DeviceInfo_family"] = np.array(
            [device_family(v) for v in df["DeviceInfo"]], dtype=object
        )
    base = df.drop(columns=["DeviceInfo"], errors="ignore")
    keyed = pd.DataFrame(
        {"uid": extra["uid"], "card1": df["card1"].to_numpy(), TIME: t, AMOUNT: amt}, index=df.index
    )
    hist_uid = entity_history_features(keyed, "uid", "UID")
    hist_card = entity_history_features(keyed, "card1", "CARD1")
    emails = email_features(df)
    return pd.concat(
        [base, pd.DataFrame(extra, index=df.index), hist_uid, hist_card, emails], axis=1
    )


class FraudFeatures:
    name = "fraud"

    def build(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return fraud_features(tables["transactions"])
