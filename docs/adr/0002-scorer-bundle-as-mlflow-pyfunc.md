# ADR 0002 – Ship the whole scoring pipeline as one MLflow pyfunc ("Scorer")

* Status: accepted
* Date: 2026-10-03

## Context
A credit decision needs: feature engineering → categorical encoding (fitted target-encoding
maps) → feature selection → model → calibration → cost-optimal threshold → risk grade →
SHAP reason codes. If serving re-implements any of these, training/serving skew is a matter
of time.

## Decision
`risklens.scorer.Scorer` holds all fitted steps and the *code path* (`prepare_input` →
`encoder.transform` → `model.predict_proba` → `calibrator` → `decide`). It is pickled with
joblib and logged as an MLflow pyfunc (`risklens.pyfunc.RiskLensPyfunc`) with a signature,
an input example and pinned library versions. Serving loads `models:/<name>@champion` and
calls `unwrap_python_model().scorer`.

The trainer itself calls `prepare_input` on the raw dataset (the features stage stores raw
application columns + history aggregates only), so the exact serving function is what
produced the training matrix.

## Consequences
* + One artifact = one version = reproducible decision (threshold and grade bands travel with the model).
* + Any MLflow consumer (batch job, notebook) gets the same behaviour via `mlflow.pyfunc.load_model`.
* − The serving image must have the `risklens` package at a compatible version
  (pinned in the pyfunc requirements; images built from the same commit).
* − Pickles are not a security boundary: only the CI pipeline can write to the registry.
