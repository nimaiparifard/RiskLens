from __future__ import annotations

import numpy as np
import pandas as pd

from risklens.features.fraud import device_family, entity_history_features, fraud_features


def _tx(rows):
    return pd.DataFrame(
        rows, columns=["TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1"]
    )


def test_velocity_counts_and_amounts():
    df = _tx(
        [
            (1, 0, 10.0, 1, 100.0, 0.0),
            (2, 1000, 20.0, 1, 100.0, 0.0),
            (3, 3000, 30.0, 1, 100.0, 0.0),
            (4, 5000, 40.0, 1, 100.0, 0.0),  # 1h window: tx at 3000 only (2000s ago)
            (5, 100, 99.0, 2, 200.0, 0.0),  # other card
        ]
    )
    h = entity_history_features(df, "card1", "C")
    assert h["C_TX_COUNT"].tolist() == [0, 1, 2, 3, 0]
    assert h["C_COUNT_1H"].tolist() == [0, 1, 2, 1, 0]
    assert h.loc[3, "C_AMT_SUM_1H"] == 30.0
    assert h.loc[3, "C_AMT_MEAN"] == 20.0  # mean of 10, 20, 30
    assert h.loc[1, "C_SECS_SINCE_LAST"] == 1000
    assert np.isnan(h.loc[0, "C_SECS_SINCE_LAST"])
    assert np.isnan(h.loc[0, "C_AMT_MEAN"])


def test_order_invariance():
    df = _tx([(1, 0, 10.0, 1, 1.0, 0.0), (2, 50, 20.0, 1, 1.0, 0.0), (3, 70, 5.0, 1, 1.0, 0.0)])
    a = entity_history_features(df, "card1", "C")
    shuffled = df.iloc[[2, 0, 1]]
    b = entity_history_features(shuffled, "card1", "C").loc[a.index]
    pd.testing.assert_frame_equal(a, b)


def test_no_lookahead(fraud_tx):
    """Appending future transactions must not change features of past ones."""
    past = fraud_tx.iloc[:1500]
    full = fraud_tx
    f_past = fraud_features(past)
    f_full = fraud_features(full).iloc[:1500]
    cols = [c for c in f_past.columns if c.startswith(("UID_", "CARD1_"))]
    pd.testing.assert_frame_equal(f_past[cols], f_full[cols])


def test_online_equals_offline(fraud_tx):
    """Serving computes features on [card history + new tx]; must equal the batch value."""
    offline = fraud_features(fraud_tx)
    i = 1800
    card = fraud_tx.loc[i, "card1"]
    history = fraud_tx.iloc[:i]
    online_input = pd.concat([history[history["card1"] == card], fraud_tx.iloc[[i]]])
    online = fraud_features(online_input).iloc[-1]
    for c in (
        "UID_TX_COUNT",
        "UID_COUNT_24H",
        "CARD1_AMT_MEAN",
        "CARD1_COUNT_1H",
        "UID_AMT_TO_MEAN",
    ):
        a, b = online[c], offline.loc[i, c]
        assert (np.isnan(a) and np.isnan(b)) or np.isclose(a, b), c


def test_email_and_device_features(fraud_tx):
    out = fraud_features(fraud_tx)
    assert {"P_emaildomain_vendor", "EMAIL_MATCH", "DeviceInfo_family", "uid", "HOUR"} <= set(
        out.columns
    )
    assert device_family("SM-G9650 Build/R16NW") == "samsung"
    assert device_family("Windows") == "windows"
    assert pd.isna(device_family(None))
