"""Data validation (pandera). The pipeline FAILS if a schema breaks.

Known anomaly: DAYS_EMPLOYED = 365243 means "unemployed / pensioner". It is
allowed by the schema here and repaired in ``features.application``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

DAYS_EMPLOYED_ANOMALY = 365243


class DataValidationError(ValueError):
    pass


def _pa() -> Any:  # lazy import: pandera is a training-only dependency
    try:
        import pandera.pandas as pa_mod
    except ImportError:  # pandera < 0.24
        import pandera as pa_mod  # type: ignore[no-redef]
    return pa_mod


def application_schema(with_target: bool = True) -> Any:
    pa = _pa()
    Check = pa.Check
    cols = {
        "SK_ID_CURR": pa.Column(int, unique=True),
        "NAME_CONTRACT_TYPE": pa.Column(str, Check.isin(["Cash loans", "Revolving loans"])),
        "CODE_GENDER": pa.Column(str, Check.isin(["M", "F", "XNA"]), nullable=True),
        "AMT_INCOME_TOTAL": pa.Column(float, Check.gt(0), coerce=True),
        "AMT_CREDIT": pa.Column(float, Check.gt(0), coerce=True),
        "AMT_ANNUITY": pa.Column(float, Check.gt(0), nullable=True, coerce=True),
        "AMT_GOODS_PRICE": pa.Column(float, Check.ge(0), nullable=True, coerce=True),
        "DAYS_BIRTH": pa.Column(int, Check.lt(0), coerce=True),
        "DAYS_EMPLOYED": pa.Column(
            int, Check.le(DAYS_EMPLOYED_ANOMALY), coerce=True, nullable=True
        ),
        "CNT_CHILDREN": pa.Column(int, Check.ge(0), coerce=True),
        "EXT_SOURCE_1": pa.Column(float, Check.in_range(0, 1), nullable=True, coerce=True),
        "EXT_SOURCE_2": pa.Column(float, Check.in_range(0, 1), nullable=True, coerce=True),
        "EXT_SOURCE_3": pa.Column(float, Check.in_range(0, 1), nullable=True, coerce=True),
    }
    if with_target:
        cols["TARGET"] = pa.Column(int, Check.isin([0, 1]), coerce=True)
    return pa.DataFrameSchema(cols, strict=False, coerce=False)


def bureau_schema() -> Any:
    pa = _pa()
    Check = pa.Check
    return pa.DataFrameSchema(
        {
            "SK_ID_CURR": pa.Column(int),
            "SK_ID_BUREAU": pa.Column(int, unique=True),
            "CREDIT_ACTIVE": pa.Column(str, Check.isin(["Active", "Closed", "Sold", "Bad debt"])),
            "DAYS_CREDIT": pa.Column(float, Check.le(0), coerce=True),
            "AMT_CREDIT_SUM": pa.Column(float, Check.ge(0), nullable=True, coerce=True),
        },
        strict=False,
    )


def fraud_schema() -> Any:
    pa = _pa()
    Check = pa.Check
    return pa.DataFrameSchema(
        {
            "TransactionID": pa.Column(int, unique=True),
            "isFraud": pa.Column(int, Check.isin([0, 1]), coerce=True),
            "TransactionDT": pa.Column(int, Check.ge(0), coerce=True),
            "TransactionAmt": pa.Column(float, Check.gt(0), coerce=True),
            "ProductCD": pa.Column(str, Check.isin(["W", "C", "R", "H", "S"])),
            "card1": pa.Column(int, coerce=True),
        },
        strict=False,
    )


def validate(df: pd.DataFrame, schema: Any, name: str) -> pd.DataFrame:
    """Validate lazily (collect ALL failures) and raise a readable error."""
    pa = _pa()
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        cases = exc.failure_cases
        summary = (
            cases.groupby(["column", "check"], dropna=False).size().reset_index(name="n_failures")
        )
        raise DataValidationError(
            f"{name}: {len(cases)} schema violations\n{summary.to_string(index=False)}"
        ) from exc


def validate_credit_tables(tables: dict[str, pd.DataFrame], with_target: bool = True) -> None:
    validate(tables["application"], application_schema(with_target), "application")
    if "bureau" in tables:
        validate(tables["bureau"], bureau_schema(), "bureau")


def validate_fraud(df: pd.DataFrame) -> None:
    validate(df, fraud_schema(), "transactions")
