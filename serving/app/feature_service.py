"""Feature services: the SAME risklens feature functions used in training.

Credit: raw application (+ optional history aggregates) -> Scorer.score
Fraud:  [this card's online history + new transaction] -> fraud_features ->
        last row -> Scorer.score_prepared. Velocity / expanding features are
        therefore computed exactly as in the offline training table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from risklens.interfaces import HistoryStore, ScoreResult
from risklens.scorer import Explain, Scorer


def _frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


class CreditFeatureService:
    def score(
        self, scorer: Scorer, features: dict[str, Any], explain: Explain = True
    ) -> ScoreResult:
        return scorer.score(_frame([features]), explain=explain)[0]

    def score_batch(
        self, scorer: Scorer, items: list[dict[str, Any]], explain: bool = False
    ) -> list[ScoreResult]:
        return scorer.score(_frame(items), explain=explain)


class FraudFeatureService:
    HISTORY_FIELDS = ("TransactionID", "TransactionDT", "TransactionAmt", "card1", "addr1", "D1")

    def __init__(self, store: HistoryStore) -> None:
        self.store = store

    def key(self, tx: dict[str, Any]) -> str:
        return str(tx["card1"])

    def score(
        self, scorer: Scorer, tx: dict[str, Any], explain: Explain = "flagged", record: bool = True
    ) -> ScoreResult:
        history = self.store.get(self.key(tx))
        frame = _frame([*history, tx])
        prepared = scorer.prepare(frame)
        result = scorer.score_prepared(prepared.iloc[[-1]], explain=explain)[0]
        if record:
            self.store.append(self.key(tx), {k: tx.get(k) for k in self.HISTORY_FIELDS})
        return result
