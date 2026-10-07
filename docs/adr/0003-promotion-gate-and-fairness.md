# ADR 0003 – Champion/challenger with a cost guard and a fairness gate

* Status: accepted
* Date: 2026-10-03

## Context
Auto-deploying whatever trains best on AUC is unsafe: a model can gain AUC while increasing
the business cost at the operating threshold, or while widening an approval-rate gap between
protected groups.

## Decision
A new registered version becomes **challenger** only if

1. `roc_auc ≥ champion.roc_auc + 0.002` (noise floor on a 46k-row test set),
2. every guard metric is not worse (`expected_cost` for credit, `recall_at_1pct_fpr` for fraud),
3. `fairness_max_dp_gap ≤ 0.10` over the **gated** protected attributes.

A human approves challenger → champion through the GitHub `production` environment.
Optionally the challenger runs in **shadow mode** first (scores every request, never decides).

### Which attributes gate
* `CODE_GENDER` is a protected attribute: **excluded from model inputs** and its decision gap
  gates promotion.
* Age (`AGE_GROUP`) is audited and reported but does **not** gate: under ECOA/Reg B age may be
  used in an empirically derived, statistically sound scoring system (as long as older
  applicants are not disfavoured – here they are favoured). The measured gap (≈0.25,
  driven by an observed 12.3% default rate under 25 vs 5.2% over 55) is documented in
  `docs/fairness.md` together with a ThresholdOptimizer mitigation and its cost.

## Consequences
* + Promotions are explainable in one table (posted as the GitHub job summary).
* + Business and fairness regressions cannot ship silently.
* − Small but real improvements (< 0.002 AUC) are rejected – acceptable for a regulated decision.
