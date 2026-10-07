"""Feature-builder registry. New data source = new module + one line here."""

from __future__ import annotations

import logging

import pandas as pd

from risklens.features.application import ApplicationFeatures, application_features
from risklens.features.bureau import BureauFeatures
from risklens.features.common import safe_div
from risklens.features.credit_card import CreditCardFeatures
from risklens.features.installments import InstallmentsFeatures
from risklens.features.pos_cash import PosCashFeatures
from risklens.features.previous import PreviousFeatures
from risklens.interfaces import FeatureBuilder

log = logging.getLogger(__name__)

FEATURE_BUILDERS: dict[str, FeatureBuilder] = {
    "application": ApplicationFeatures(),
    "bureau": BureauFeatures(),
    "previous": PreviousFeatures(),
    "installments": InstallmentsFeatures(),
    "pos_cash": PosCashFeatures(),
    "credit_card": CreditCardFeatures(),
}

# source table each aggregate builder needs
_REQUIRES = {
    "bureau": "bureau",
    "previous": "previous",
    "installments": "installments",
    "pos_cash": "pos_cash",
    "credit_card": "credit_card",
}

KEY = "SK_ID_CURR"


def build_credit_features(tables: dict[str, pd.DataFrame], sources: list[str]) -> pd.DataFrame:
    """RAW application row LEFT JOIN every per-customer history aggregate.

    Application-level features are deliberately NOT applied here: the trainer
    applies ``prepare_credit_input`` - the exact function serving uses - to
    this frame, so offline and online feature code can never diverge.
    """
    base = tables["application"]
    parts = []
    for name in sources:
        if name == "application":
            continue
        if _REQUIRES.get(name) not in tables:
            log.warning("source '%s' unavailable – skipped", name)
            continue
        log.info("building %s features", name)
        part = FEATURE_BUILDERS[name].build(tables)
        log.info("  -> %d customers x %d features", *part.shape)
        parts.append(part)
    out = base.set_index(KEY)
    if parts:
        out = out.join(parts, how="left")
    return out.copy().reset_index()


def finalize_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-table features (need application + history columns). Pure."""
    out = df.copy()
    income = (
        pd.to_numeric(out.get("AMT_INCOME_TOTAL"), errors="coerce")
        if "AMT_INCOME_TOTAL" in out
        else None
    )
    if income is not None:
        if "BURO_AMT_CREDIT_SUM_DEBT_SUM" in out:
            out["BURO_DEBT_TO_INCOME"] = safe_div(
                out["BURO_AMT_CREDIT_SUM_DEBT_SUM"], income
            ).astype("float32")
        if "INSTAL_AMT_INSTALMENT_MEAN" in out:
            out["INSTAL_TO_INCOME"] = safe_div(out["INSTAL_AMT_INSTALMENT_MEAN"], income).astype(
                "float32"
            )
    return out


def prepare_credit_input(raw: pd.DataFrame) -> pd.DataFrame:
    """Serving-time path: raw application fields (+ optional pre-computed history
    aggregates) -> the exact same columns the model saw in training."""
    return finalize_credit_features(application_features(raw))


def history_feature_names(features: pd.DataFrame, application_columns: list[str]) -> list[str]:
    """Columns that come from history tables (must be supplied at serving time)."""
    return [c for c in features.columns if c not in set(application_columns)]
