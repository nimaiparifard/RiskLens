"""Small, pure, vectorised helpers shared by all feature modules."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

DAYS_SENTINEL = 365243  # Home Credit's "not applicable / infinite" day value


def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    """a / b that returns NaN instead of +-inf for zero / missing denominators."""
    a_arr = np.asarray(a, dtype="float64")
    b_arr = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = a_arr / b_arr
    out[~np.isfinite(out)] = np.nan
    return out


def replace_sentinel(
    df: pd.DataFrame, cols: Iterable[str], value: float = DAYS_SENTINEL
) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].where(df[c] != value)
    return df


def aggregate(
    df: pd.DataFrame,
    by: str,
    spec: dict[str, list[str]],
    prefix: str,
) -> pd.DataFrame:
    """groupby(by).agg(spec) with flat ``PREFIX_COL_AGG`` column names.

    Columns missing from ``df`` are silently skipped so the same spec works on
    partial / synthetic tables.
    """
    spec = {c: aggs for c, aggs in spec.items() if c in df.columns}
    if df.empty or not spec:
        return pd.DataFrame(index=pd.Index([], name=by))
    out = df.groupby(by).agg(spec)
    out.columns = [f"{prefix}_{c}_{agg}".upper() for c, agg in out.columns]
    return out.astype("float32")


def group_slope(df: pd.DataFrame, by: str, x: str, y: str) -> pd.Series:
    """Least-squares slope of y on x per group, fully vectorised (trend features).

    slope = (n*Sxy - Sx*Sy) / (n*Sxx - Sx^2); NaN for groups with < 2 points.
    """
    d = df[[by, x, y]].dropna()
    d = d.assign(_xy=d[x] * d[y], _xx=d[x] * d[x])
    g = d.groupby(by).agg(
        n=(x, "size"), sx=(x, "sum"), sy=(y, "sum"), sxy=("_xy", "sum"), sxx=("_xx", "sum")
    )
    num = g["n"] * g["sxy"] - g["sx"] * g["sy"]
    den = g["n"] * g["sxx"] - g["sx"] ** 2
    slope = pd.Series(safe_div(num, den), index=g.index)
    slope[g["n"] < 2] = np.nan
    return slope


def share_by_category(
    df: pd.DataFrame, by: str, col: str, prefix: str, values: list[str]
) -> pd.DataFrame:
    """Share of rows per group having ``col == value`` for each value."""
    out = pd.DataFrame(index=pd.Index(df[by].unique(), name=by))
    if col not in df.columns:
        return out
    grouped = df.groupby(by)[col]
    for v in values:
        key = str(v).upper().replace(" ", "_").replace("/", "_")
        out[f"{prefix}_{col}_{key}_SHARE".upper()] = grouped.apply(lambda s, v=v: (s == v).mean())
    return out.astype("float32")


def rate_by_category(
    df: pd.DataFrame, by: str, col: str, prefix: str, values: list[str]
) -> pd.DataFrame:
    """Vectorised version of ``share_by_category`` (one-hot + groupby mean)."""
    if col not in df.columns or df.empty:
        return pd.DataFrame(index=pd.Index([], name=by))
    dummies = pd.DataFrame({by: df[by].to_numpy()})
    for v in values:
        key = str(v).upper().replace(" ", "_").replace("/", "_")
        dummies[f"{prefix}_{col}_{key}_SHARE".upper()] = (df[col].to_numpy() == v).astype("float32")
    return dummies.groupby(by).mean().astype("float32")
