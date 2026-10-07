"""Business cost, cost-optimal thresholds and risk grades.

Credit: a positive (default) flagged -> loan rejected.
  FN (approved defaulter) costs amount * LGD
  FP (rejected good)      costs amount * margin  (+ fixed)
Fraud: positive flagged -> transaction blocked.
  FN (missed fraud)       costs amount * fn_rate
  FP (blocked genuine)    costs fixed friction
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.config import CostConfig


def amounts_for(df: pd.DataFrame | None, cfg: CostConfig, n: int) -> np.ndarray:
    if df is not None and cfg.amount_column and cfg.amount_column in df:
        a = pd.to_numeric(df[cfg.amount_column], errors="coerce").to_numpy(dtype="float64")
        return np.where(np.isfinite(a), a, cfg.default_amount)
    return np.full(n, cfg.default_amount)


def expected_cost(
    y: np.ndarray, p: np.ndarray, threshold: float, amounts: np.ndarray, cfg: CostConfig
) -> float:
    y = np.asarray(y).astype(bool)
    flag = np.asarray(p) >= threshold
    fn = ~flag & y
    fp = flag & ~y
    return float(
        (amounts[fn] * cfg.fn_cost_rate).sum()
        + (amounts[fp] * cfg.fp_cost_rate).sum()
        + fp.sum() * cfg.fp_cost_fixed
    )


def baseline_cost(y: np.ndarray, amounts: np.ndarray, cfg: CostConfig) -> float:
    """Baseline policy = approve everything (no model)."""
    return expected_cost(y, np.zeros(len(y)), 1.0, amounts, cfg)


def optimal_threshold(
    y: np.ndarray, p: np.ndarray, amounts: np.ndarray, cfg: CostConfig, n_grid: int = 400
) -> tuple[float, float]:
    """Threshold minimising expected cost on the given (validation) set."""
    grid = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_grid)))
    costs = np.array([expected_cost(y, p, t, amounts, cfg) for t in grid])
    best = int(np.argmin(costs))
    return float(grid[best]), float(costs[best])


def assign_grades(p: np.ndarray, bands: dict[str, float]) -> np.ndarray:
    """Map PD -> grade using upper bounds (A <= 3%, B <= 6%, ...)."""
    labels = list(bands.keys())
    uppers = np.array(list(bands.values()), dtype="float64")
    idx = np.searchsorted(uppers, np.asarray(p, dtype="float64"), side="left")
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.array(labels, dtype=object)[idx]


def grade_table(y: np.ndarray, p: np.ndarray, bands: dict[str, float]) -> pd.DataFrame:
    grades = assign_grades(p, bands)
    df = pd.DataFrame({"grade": grades, "y": np.asarray(y), "p": np.asarray(p)})
    out = df.groupby("grade").agg(
        n=("y", "size"), default_rate=("y", "mean"), mean_pd=("p", "mean")
    )
    out["share"] = out["n"] / out["n"].sum()
    return out.reindex(list(bands.keys())).reset_index()


def business_summary(
    y: np.ndarray, p: np.ndarray, threshold: float, amounts: np.ndarray, cfg: CostConfig
) -> dict[str, float]:
    cost = expected_cost(y, p, threshold, amounts, cfg)
    base = baseline_cost(y, amounts, cfg)
    flag = np.asarray(p) >= threshold
    return {
        "threshold": float(threshold),
        "expected_cost": cost,
        "baseline_cost": base,
        "loss_saved": base - cost,
        "loss_saved_pct": (base - cost) / base if base else 0.0,
        "approval_rate": float(1 - flag.mean()),
        "flag_rate": float(flag.mean()),
        "bad_rate_approved": float(np.asarray(y)[~flag].mean()) if (~flag).any() else 0.0,
    }


def approval_rate_at_risk(y: np.ndarray, p: np.ndarray, max_bad_rate: float) -> float:
    """Highest approval rate whose approved book keeps bad rate <= ``max_bad_rate``."""
    order = np.argsort(p)
    y_sorted = np.asarray(y)[order]
    cum_bad = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    ok = np.flatnonzero(cum_bad <= max_bad_rate)
    return float((int(ok[-1]) + 1) / len(y_sorted)) if len(ok) else 0.0
