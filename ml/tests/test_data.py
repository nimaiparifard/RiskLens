from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.config import load_config
from risklens.data.schemas import DataValidationError, validate_credit_tables, validate_fraud
from risklens.data.split import TEST, TRAIN, VALID, ordered_split, stratified_split, time_split

from .conftest import CONFIGS


def test_synthetic_credit_passes_schema(credit_tables):
    validate_credit_tables(credit_tables)


def test_schema_rejects_bad_rows(credit_tables):
    bad = {k: v.copy() for k, v in credit_tables.items()}
    bad["application"].loc[0, "AMT_INCOME_TOTAL"] = -5
    bad["application"].loc[1, "NAME_CONTRACT_TYPE"] = "Mortgage"
    with pytest.raises(DataValidationError) as exc:
        validate_credit_tables(bad)
    assert "AMT_INCOME_TOTAL" in str(exc.value) and "NAME_CONTRACT_TYPE" in str(exc.value)


def test_schema_rejects_duplicate_ids(credit_tables):
    bad = {k: v.copy() for k, v in credit_tables.items()}
    bad["application"].loc[1, "SK_ID_CURR"] = bad["application"].loc[0, "SK_ID_CURR"]
    with pytest.raises(DataValidationError):
        validate_credit_tables(bad)


def test_fraud_schema(fraud_tx):
    validate_fraud(fraud_tx)
    bad = fraud_tx.copy()
    bad.loc[0, "TransactionAmt"] = 0
    with pytest.raises(DataValidationError):
        validate_fraud(bad)


def test_stratified_split_proportions(credit_tables):
    app = credit_tables["application"]
    s = stratified_split(app["SK_ID_CURR"], app["TARGET"], 0.15, 0.15, seed=0)
    frac = s["split"].value_counts(normalize=True)
    assert abs(frac[TRAIN] - 0.70) < 0.01 and abs(frac[TEST] - 0.15) < 0.01
    rates = app["TARGET"].groupby(s["split"].to_numpy()).mean()
    assert rates.max() - rates.min() < 0.03
    # deterministic
    s2 = stratified_split(app["SK_ID_CURR"], app["TARGET"], 0.15, 0.15, seed=0)
    assert (s["split"] == s2["split"]).all()


def test_time_split_is_ordered(fraud_tx):
    s = time_split(fraud_tx["TransactionID"], fraud_tx["TransactionDT"], 0.1, 0.2)
    t = fraud_tx["TransactionDT"].to_numpy()
    sp = s["split"].to_numpy()
    assert t[sp == TRAIN].max() <= t[sp == VALID].min()
    assert t[sp == VALID].max() <= t[sp == TEST].min()


def test_ordered_split():
    ids = pd.Series(np.arange(100))
    part = ordered_split(ids, 0.2)
    assert (part[ids >= 80] == TEST).all() and (part[ids < 80] == TRAIN).all()


def test_config_loading_and_overrides():
    cfg = load_config(CONFIGS / "credit.yaml", ["data.nrows=1000", "tuning.enabled=true"])
    assert cfg.data.nrows == 1000 and cfg.tuning.enabled
    assert cfg.costs.fn_cost_rate == 0.45  # merged from costs.yaml
    assert cfg.final_model == "stacking"
    fraud = load_config(CONFIGS / "fraud.yaml")
    assert fraud.costs.target_fpr == 0.01 and fraud.split.strategy == "time"


def test_config_rejects_unknown_final_model():
    with pytest.raises(ValueError):
        load_config(CONFIGS / "credit.yaml", ["final_model=does_not_exist"])
