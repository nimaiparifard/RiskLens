# ADR 0004 – Fraud: explain only flagged transactions, keep the big model

* Status: accepted
* Date: 2026-10-07

## Context
The fraud authorisation call has a 50 ms budget. The champion (LightGBM, 1,017 trees ×
256 leaves, unlimited depth) predicts in ~3 ms, but exact TreeSHAP on it takes ~214 ms per
transaction. The first serving version explained every transaction: **p95 ≈ 300 ms**,
breaking the budget.

Measured alternatives (same features, same time split):

| Leaves / depth | ROC-AUC | Recall @ 1% FPR | TreeSHAP |
|---|---|---|---|
| 256 / unlimited | 0.9059 | 47.9% | 214 ms |
| 128 / 12 | 0.9013 | 47.5% | 72 ms |
| 64 / 10 | 0.8993 | 43.9% | 22 ms |

## Decision
1. **Keep the 256-leaf champion.** Shrinking to 64 leaves would cut fraud recall by 4 points
   (~8% more missed fraud) only to make *explanations* cheaper.
2. **`explain="flagged"` is the default for fraud**: SHAP runs only for `review` / `block`
   decisions (~3% of traffic), the transactions an analyst actually reads. Approved
   transactions return no reasons.
3. Plain prediction uses one thread for single rows (avoids OpenMP overhead);
   TreeSHAP may use all cores.

Result (live, one worker): authorisation path **p50 13 ms / p95 18 ms**.

## Consequences
* + The 50 ms budget holds for the decision path.
* − Flagged transactions still take ~215 ms; acceptable because they go to a review queue or
  are blocked anyway, but a blocked customer waits longer.
* Next step: return the decision immediately and compute reasons asynchronously
  (`/v1/fraud/explain` called by a Celery task), or evaluate the 128-leaf model in shadow mode.
