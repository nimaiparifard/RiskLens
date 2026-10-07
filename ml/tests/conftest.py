from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from risklens.data.synthetic import make_credit_tables, make_fraud_transactions

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture(scope="session")
def credit_tables() -> dict[str, pd.DataFrame]:
    return make_credit_tables(n=600, seed=1)


@pytest.fixture(scope="session")
def fraud_tx() -> pd.DataFrame:
    return make_fraud_transactions(n=2000, seed=1)


@pytest.fixture
def tiny_application() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3],
            "AMT_INCOME_TOTAL": [100_000.0, 0.0, 50_000.0],
            "AMT_CREDIT": [300_000.0, 200_000.0, 50_000.0],
            "AMT_ANNUITY": [15_000.0, 10_000.0, None],
            "AMT_GOODS_PRICE": [270_000.0, 0.0, 50_000.0],
            "DAYS_BIRTH": [-10_000, -20_000, -8_000],
            "DAYS_EMPLOYED": [-1_000, 365243, -500],
            "CNT_FAM_MEMBERS": [2.0, 1.0, 0.0],
            "CNT_CHILDREN": [0, 0, 1],
            "CODE_GENDER": ["F", "XNA", "M"],
            "EXT_SOURCE_1": [0.5, None, 0.2],
            "EXT_SOURCE_2": [0.6, 0.3, None],
            "EXT_SOURCE_3": [None, None, None],
            "FLAG_DOCUMENT_3": [1, 0, 1],
            "FLAG_DOCUMENT_5": [1, 0, 0],
        }
    )
