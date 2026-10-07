"""Synthetic Home-Credit-like and IEEE-like data.

Used by unit tests, the CI smoke-train and anyone without Kaggle access.
The target depends on the same drivers as the real data (external scores,
debt burden, late payments) so models learn something non-trivial.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ORGS = ["Business Entity Type 3", "School", "Government", "Self-employed", "Medicine", "Other"]
OCCUPATIONS = ["Laborers", "Core staff", "Accountants", "Managers", "Drivers", "Sales staff", None]
INCOME_TYPES = ["Working", "Commercial associate", "Pensioner", "State servant"]
EDUCATION = ["Secondary / secondary special", "Higher education", "Incomplete higher"]
FAMILY = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_credit_tables(n: int = 2000, seed: int = 0) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ids = np.arange(100_000, 100_000 + n)
    income = rng.lognormal(11.9, 0.5, n).round(0)
    credit = (income * rng.uniform(1, 6, n)).round(0)
    annuity = (credit / rng.uniform(10, 40, n)).round(1)
    goods = (credit * rng.uniform(0.8, 1.0, n)).round(0)
    days_birth = -rng.integers(21 * 365, 68 * 365, n)
    days_emp = -rng.integers(30, 15 * 365, n)
    pensioner = rng.random(n) < 0.15
    days_emp = np.where(pensioner, 365243, days_emp)
    ext = rng.beta(4, 3, (n, 3))
    ext[rng.random((n, 3)) < np.array([0.55, 0.01, 0.2])] = np.nan
    late_rate = rng.beta(1, 12, n)

    logit = (
        -2.6
        - 2.2 * (np.nan_to_num(ext[:, 1], nan=0.5) - 0.5) * 3
        - 1.2 * (np.nan_to_num(ext[:, 2], nan=0.5) - 0.5) * 3
        + 0.6 * np.log(credit / income)
        + 6.0 * late_rate
        + 0.3 * (days_birth > -30 * 365)
    )
    target = (rng.random(n) < _sigmoid(logit)).astype(int)

    app = pd.DataFrame(
        {
            "SK_ID_CURR": ids,
            "TARGET": target,
            "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n, p=[0.9, 0.1]),
            "CODE_GENDER": rng.choice(["F", "M"], n, p=[0.65, 0.35]),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], n),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], n),
            "CNT_CHILDREN": rng.poisson(0.4, n),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "NAME_INCOME_TYPE": np.where(pensioner, "Pensioner", rng.choice(INCOME_TYPES[:2], n)),
            "NAME_EDUCATION_TYPE": rng.choice(EDUCATION, n),
            "NAME_FAMILY_STATUS": rng.choice(FAMILY, n),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.001, 0.07, n),
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_emp,
            "DAYS_REGISTRATION": -rng.integers(0, 20 * 365, n).astype(float),
            "DAYS_ID_PUBLISH": -rng.integers(0, 6000, n),
            "DAYS_LAST_PHONE_CHANGE": -rng.integers(0, 3000, n).astype(float),
            "OWN_CAR_AGE": np.where(rng.random(n) < 0.6, np.nan, rng.integers(0, 30, n)),
            "CNT_FAM_MEMBERS": rng.integers(1, 5, n).astype(float),
            "REGION_RATING_CLIENT": rng.choice([1, 2, 3], n, p=[0.1, 0.75, 0.15]),
            "OCCUPATION_TYPE": rng.choice(np.array(OCCUPATIONS, dtype=object), n),
            "ORGANIZATION_TYPE": np.where(pensioner, "XNA", rng.choice(ORGS, n)),
            "EXT_SOURCE_1": ext[:, 0],
            "EXT_SOURCE_2": ext[:, 1],
            "EXT_SOURCE_3": ext[:, 2],
            "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
            "FLAG_DOCUMENT_6": rng.integers(0, 2, n),
            "FLAG_DOCUMENT_8": rng.integers(0, 2, n),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.poisson(1.5, n).astype(float),
        }
    )

    # ---- bureau + bureau_balance
    n_b = rng.poisson(3, n)
    b_ids = np.repeat(ids, n_b)
    nb = len(b_ids)
    bureau = pd.DataFrame(
        {
            "SK_ID_CURR": b_ids,
            "SK_ID_BUREAU": np.arange(5_000_000, 5_000_000 + nb),
            "CREDIT_ACTIVE": rng.choice(["Active", "Closed"], nb, p=[0.4, 0.6]),
            "CREDIT_CURRENCY": "currency 1",
            "DAYS_CREDIT": -rng.integers(1, 2900, nb).astype(float),
            "CREDIT_DAY_OVERDUE": np.where(rng.random(nb) < 0.03, rng.integers(1, 90, nb), 0),
            "DAYS_CREDIT_ENDDATE": rng.integers(-2900, 2000, nb).astype(float),
            "DAYS_ENDDATE_FACT": np.where(rng.random(nb) < 0.5, np.nan, -rng.integers(1, 2900, nb)),
            "AMT_CREDIT_MAX_OVERDUE": np.where(
                rng.random(nb) < 0.7, np.nan, rng.exponential(3000, nb)
            ),
            "CNT_CREDIT_PROLONG": rng.poisson(0.01, nb),
            "AMT_CREDIT_SUM": rng.lognormal(12, 1, nb),
            "AMT_CREDIT_SUM_DEBT": rng.lognormal(10, 1.5, nb) * (rng.random(nb) < 0.6),
            "AMT_CREDIT_SUM_LIMIT": np.where(rng.random(nb) < 0.6, np.nan, 0.0),
            "AMT_CREDIT_SUM_OVERDUE": np.where(rng.random(nb) < 0.02, rng.exponential(5000, nb), 0),
            "CREDIT_TYPE": rng.choice(["Consumer credit", "Credit card", "Car loan"], nb),
            "DAYS_CREDIT_UPDATE": -rng.integers(0, 1000, nb),
            "AMT_ANNUITY": np.where(rng.random(nb) < 0.7, np.nan, rng.lognormal(9, 1, nb)),
        }
    )
    m_bb = rng.integers(1, 6, nb)
    bb_ids = np.repeat(bureau["SK_ID_BUREAU"].to_numpy(), m_bb)
    bureau_balance = pd.DataFrame(
        {
            "SK_ID_BUREAU": bb_ids,
            "MONTHS_BALANCE": -np.concatenate([np.arange(m) for m in m_bb]),
            "STATUS": rng.choice(
                ["0", "1", "2", "C", "X"], len(bb_ids), p=[0.5, 0.05, 0.02, 0.33, 0.1]
            ),
        }
    )

    # ---- previous applications
    n_p = rng.poisson(2.5, n)
    p_curr = np.repeat(ids, n_p)
    npv = len(p_curr)
    prev_ids = np.arange(2_000_000, 2_000_000 + npv)
    amt_app = rng.lognormal(11, 1, npv)
    status = rng.choice(
        ["Approved", "Refused", "Canceled", "Unused offer"], npv, p=[0.62, 0.18, 0.18, 0.02]
    )
    previous = pd.DataFrame(
        {
            "SK_ID_PREV": prev_ids,
            "SK_ID_CURR": p_curr,
            "NAME_CONTRACT_TYPE": rng.choice(
                ["Consumer loans", "Cash loans", "Revolving loans"], npv
            ),
            "AMT_ANNUITY": amt_app / rng.uniform(6, 30, npv),
            "AMT_APPLICATION": amt_app,
            "AMT_CREDIT": np.where(status == "Approved", amt_app * rng.uniform(0.9, 1.2, npv), 0.0),
            "AMT_DOWN_PAYMENT": np.where(rng.random(npv) < 0.5, np.nan, amt_app * 0.1),
            "AMT_GOODS_PRICE": amt_app,
            "NAME_CONTRACT_STATUS": status,
            "DAYS_DECISION": -rng.integers(1, 2900, npv),
            "CNT_PAYMENT": rng.choice([6.0, 12.0, 24.0, 36.0], npv),
            "DAYS_FIRST_DUE": np.where(
                rng.random(npv) < 0.1, 365243.0, -rng.integers(1, 2800, npv)
            ),
            "DAYS_LAST_DUE": np.where(rng.random(npv) < 0.4, 365243.0, -rng.integers(1, 2800, npv)),
            "DAYS_TERMINATION": np.where(
                rng.random(npv) < 0.4, 365243.0, -rng.integers(1, 2800, npv)
            ),
            "NAME_YIELD_GROUP": rng.choice(["low_normal", "middle", "high", "XNA"], npv),
        }
    )

    # ---- installments (late-payment behaviour drives the target)
    late_by_curr = dict(zip(ids, late_rate, strict=True))
    n_i = rng.integers(1, 8, npv)
    i_prev = np.repeat(prev_ids, n_i)
    i_curr = np.repeat(p_curr, n_i)
    ni = len(i_prev)
    days_inst = -rng.integers(1, 2800, ni).astype(float)
    late = rng.random(ni) < np.array([late_by_curr[c] for c in i_curr])
    amt_inst = rng.lognormal(9, 1, ni)
    installments = pd.DataFrame(
        {
            "SK_ID_PREV": i_prev,
            "SK_ID_CURR": i_curr,
            "NUM_INSTALMENT_VERSION": 1.0,
            "NUM_INSTALMENT_NUMBER": np.concatenate([np.arange(1, k + 1) for k in n_i]),
            "DAYS_INSTALMENT": days_inst,
            "DAYS_ENTRY_PAYMENT": days_inst
            + np.where(late, rng.integers(1, 60, ni), -rng.integers(0, 10, ni)),
            "AMT_INSTALMENT": amt_inst,
            "AMT_PAYMENT": amt_inst * np.where(late, rng.uniform(0.3, 1.0, ni), 1.0),
        }
    )

    # ---- POS cash + credit card monthly snapshots
    n_m = rng.integers(1, 6, npv)
    m_prev = np.repeat(prev_ids, n_m)
    m_curr = np.repeat(p_curr, n_m)
    nm = len(m_prev)
    months = -np.concatenate([np.arange(1, k + 1) for k in n_m])
    pos_cash = pd.DataFrame(
        {
            "SK_ID_PREV": m_prev,
            "SK_ID_CURR": m_curr,
            "MONTHS_BALANCE": months,
            "CNT_INSTALMENT": rng.choice([6.0, 12.0, 24.0], nm),
            "CNT_INSTALMENT_FUTURE": rng.integers(0, 24, nm).astype(float),
            "NAME_CONTRACT_STATUS": rng.choice(
                ["Active", "Completed", "Signed"], nm, p=[0.85, 0.1, 0.05]
            ),
            "SK_DPD": np.where(rng.random(nm) < 0.05, rng.integers(1, 100, nm), 0),
            "SK_DPD_DEF": np.where(rng.random(nm) < 0.02, rng.integers(1, 30, nm), 0),
        }
    )
    limit = rng.choice([45000.0, 90000.0, 135000.0, 225000.0], nm)
    credit_card = pd.DataFrame(
        {
            "SK_ID_PREV": m_prev,
            "SK_ID_CURR": m_curr,
            "MONTHS_BALANCE": months,
            "AMT_BALANCE": limit * rng.beta(2, 3, nm),
            "AMT_CREDIT_LIMIT_ACTUAL": limit,
            "AMT_DRAWINGS_ATM_CURRENT": np.where(
                rng.random(nm) < 0.7, 0.0, rng.exponential(5000, nm)
            ),
            "AMT_DRAWINGS_CURRENT": rng.exponential(8000, nm),
            "AMT_INST_MIN_REGULARITY": limit * 0.05,
            "AMT_PAYMENT_TOTAL_CURRENT": rng.exponential(7000, nm),
            "AMT_TOTAL_RECEIVABLE": limit * rng.beta(2, 3, nm),
            "CNT_DRAWINGS_CURRENT": rng.poisson(2, nm),
            "NAME_CONTRACT_STATUS": "Active",
            "SK_DPD": np.where(rng.random(nm) < 0.03, rng.integers(1, 60, nm), 0),
            "SK_DPD_DEF": np.where(rng.random(nm) < 0.01, rng.integers(1, 30, nm), 0),
        }
    )
    return {
        "application": app,
        "bureau": bureau,
        "bureau_balance": bureau_balance,
        "previous": previous,
        "installments": installments,
        "pos_cash": pos_cash,
        "credit_card": credit_card,
    }


def make_fraud_transactions(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    """IEEE-CIS-like transactions sorted by TransactionDT (seconds)."""
    rng = np.random.default_rng(seed)
    n_cards = max(n // 8, 10)
    card1 = rng.integers(1000, 18000, n_cards)
    addr1 = rng.choice([204.0, 299.0, 325.0, 330.0, 441.0, np.nan], n_cards)
    first_day = rng.integers(0, 120, n_cards)
    card_risky = rng.random(n_cards) < 0.06
    which = rng.integers(0, n_cards, n)
    dt = np.sort(rng.integers(86400, 86400 * 180, n))
    amount = np.round(rng.lognormal(4, 1.1, n), 2)
    risky = card_risky[which]
    # fraudsters: bigger amounts, bursts, risky email domains
    amount = np.where(risky & (rng.random(n) < 0.5), amount * 4, amount)
    p_email = np.where(
        risky & (rng.random(n) < 0.6),
        rng.choice(["protonmail.com", "mail.com", "outlook.es"], n),
        rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", None], n),
    )
    logit = -5.0 + 3.2 * risky + 0.25 * np.log1p(amount / 100)
    is_fraud = (rng.random(n) < _sigmoid(logit)).astype(int)
    day = dt // 86400
    df = pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n),
            "isFraud": is_fraud,
            "TransactionDT": dt,
            "TransactionAmt": amount,
            "ProductCD": rng.choice(["W", "C", "R", "H", "S"], n, p=[0.74, 0.12, 0.06, 0.05, 0.03]),
            "card1": card1[which],
            "card2": rng.choice([111.0, 321.0, 490.0, 555.0, np.nan], n),
            "card3": 150.0,
            "card4": rng.choice(
                ["visa", "mastercard", "american express", "discover"], n, p=[0.65, 0.3, 0.03, 0.02]
            ),
            "card5": rng.choice([226.0, 224.0, 166.0, 102.0], n),
            "card6": rng.choice(["debit", "credit"], n, p=[0.75, 0.25]),
            "addr1": addr1[which],
            "addr2": 87.0,
            "dist1": np.where(rng.random(n) < 0.6, np.nan, rng.exponential(50, n)),
            "P_emaildomain": p_email,
            "R_emaildomain": np.where(rng.random(n) < 0.75, None, p_email),
            "C1": rng.poisson(1 + 4 * risky),
            "C2": rng.poisson(1 + 3 * risky),
            "C13": rng.poisson(10, n),
            "C14": rng.poisson(1, n),
            "D1": (day - first_day[which]).clip(0).astype(float),
            "D4": np.where(rng.random(n) < 0.3, np.nan, rng.integers(0, 600, n)),
            "D10": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 600, n)),
            "D15": np.where(rng.random(n) < 0.1, np.nan, rng.integers(0, 600, n)),
            "M4": rng.choice(["M0", "M1", "M2", None], n),
            "M6": rng.choice(["T", "F", None], n),
            "V258": rng.choice([1.0, 2.0, 3.0, np.nan], n),
            "V294": rng.poisson(0.5 + risky, n).astype(float),
            "id_19": np.where(rng.random(n) < 0.75, np.nan, rng.integers(100, 600, n)),
            "id_20": np.where(rng.random(n) < 0.75, np.nan, rng.integers(100, 600, n)),
            "DeviceType": rng.choice(["desktop", "mobile", None], n, p=[0.15, 0.1, 0.75]),
            "DeviceInfo": rng.choice(
                ["Windows", "iOS Device", "MacOS", "SM-G9650 Build/R16NW", "Trident/7.0", None], n
            ),
        }
    )
    return df
