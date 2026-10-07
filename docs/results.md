# Results

All numbers are produced by the code in this repo on the full Kaggle data
(`make features && make train && make train-fraud`) and are reproducible: two
independent credit runs gave identical metrics to 4 decimals (seeded splits,
OOF folds, models). Hardware: 12-core Windows desktop, 32 GB RAM, CPU only.

## Credit risk — Home Credit Default Risk

Frozen, DVC-tracked stratified split: 215,257 train / 46,127 valid / 46,127 test (8.07% defaults).
396 encoded features → 300 selected (missing / constant / collinear filters + LightGBM gain).

| Model | ROC-AUC | PR-AUC | KS | Gini | Brier | Expected cost | Recall @ 1% FPR | Fit time |
|---|---|---|---|---|---|---|---|---|
| Logistic regression (WOE scorecard) | 0.7625 | 0.2350 | 0.3973 | 0.5251 | 0.0677 | 771.8 M | 7.5% | 3 s |
| LightGBM | 0.7923 | 0.2781 | 0.4425 | 0.5846 | 0.0654 | 718.7 M | 10.7% | 24 s |
| XGBoost | 0.7907 | 0.2802 | 0.4408 | 0.5814 | 0.0654 | 725.3 M | 11.2% | 152 s |
| CatBoost | 0.7907 | 0.2795 | 0.4375 | 0.5813 | 0.0655 | 723.3 M | 10.7% | 53 s |
| **Stacking (LogReg meta on 5-fold OOF)** | **0.7933** | **0.2851** | **0.4450** | **0.5867** | **0.0653** | **717.5 M** | 11.0% | 744 s |

Every model is isotonic-calibrated on the validation set and evaluated at its own
cost-optimal threshold (chosen on validation, applied to test).

### Business view (final model, test set)
| | |
|---|---|
| Decision threshold (calibrated PD) | 0.167 |
| Approval rate | **86.2%** |
| Bad rate of approved loans | 4.9% (vs. 8.1% overall) |
| Expected loss, approve-everyone baseline | 933.8 M |
| Expected loss, RiskLens policy | 717.5 M |
| **Loss saved** | **216.2 M (−23.2%)** |
| Max approval rate keeping the approved book's bad rate ≤ 5% | 86.7% |

Cost model (`ml/configs/costs.yaml`): approving a defaulter costs `AMT_CREDIT × 45% LGD`;
rejecting a good customer costs `AMT_CREDIT × 8%` lost margin.

### Robustness
* **Out-of-time style check** – LightGBM trained on the lowest 85% of `SK_ID_CURR`, tested on
  the newest 15%: **AUC 0.7923** (same as the random split → no temporal degradation visible).
* **Calibration** – Brier 0.0653; mean predicted PD 8.05% vs. observed 8.07%
  (`reports/credit/plots/calibration_before_after.png`).
* **Stacking gain is small (+0.001 AUC over LightGBM)** at 30× the training time. It is kept
  because it also lowers expected cost; a LightGBM-only champion would be a reasonable
  simplification and the config makes it a one-line change (`final_model: lightgbm`).

### Grades (test set)
Risk grades come from calibrated PD bands (A ≤ 3%, B ≤ 6%, C ≤ 12%, D ≤ 25%, E > 25%):

| Grade | Share of applicants | Mean predicted PD | Observed default rate |
|---|---|---|---|
| A | 41.6% | 1.8% | 1.9% |
| B | 16.4% | 4.7% | 4.6% |
| C | 20.7% | 8.4% | 8.0% |
| D | 14.5% | 16.3% | 16.9% |
| E | 6.8% | 35.5% | 35.2% |

Default rates rise monotonically and match the calibrated PD within ~0.5 points in every band.

## Fraud — IEEE-CIS Fraud Detection

**Time-based** split (590,540 transactions, 182 days, 3.50% fraud):
train = first 70% of time (413,378) · valid = next 10% (59,054) · test = last 20% (118,108).

| Model | ROC-AUC | PR-AUC | KS | Brier | Recall @ 1% FPR |
|---|---|---|---|---|---|
| LightGBM (256 leaves) + reconstructed client-id velocity features | **0.9059** | **0.5356** | 0.6678 | 0.0215 | **47.9%** |

The operating threshold is set at 1% false-positive rate on validation. Expected loss falls
**47.4%** vs. letting every transaction through (`fn = amount`, `fp = 5` friction cost).

### Tree size vs. explanation cost (ADR 0004)
| Leaves / max depth | ROC-AUC | PR-AUC | Recall @ 1% FPR | TreeSHAP per transaction |
|---|---|---|---|---|
| 256 / unlimited (**champion**) | 0.9059 | 0.5356 | **47.9%** | 214 ms |
| 128 / 12 | 0.9013 | 0.5355 | 47.5% | 72 ms |
| 64 / 10 | 0.8993 | 0.5117 | 43.9% | 22 ms |

## Serving performance

Measured with `scripts/benchmark_serving.py` against one uvicorn worker on this machine,
real Kaggle rows, HTTP over localhost:

| Endpoint | serving p50 | serving p95 | HTTP p95 |
|---|---|---|---|
| `POST /v1/credit/score` (4-model stack + SHAP top-5 reasons) | 79 ms | 95 ms | 102 ms |
| **`POST /v1/fraud/score` (authorisation path; SHAP only if flagged)** | **13 ms** | **18 ms** | **24 ms** |
| `POST /v1/fraud/score` with SHAP forced on every call | 225 ms | 235 ms | 242 ms |

Throughput (concurrency 8, single worker, CPU-bound): **credit ≈ 10 req/s, fraud ≈ 52 req/s**.
Scale-out is horizontal (more uvicorn workers / Kubernetes HPA in `infra/helm`).

Live end-to-end through Django → serving (seeded stack, `scripts/smoke_e2e.py`):
fraud check 12–15 ms inside serving; credit score 80–95 ms inside serving.

## How to reproduce
```bash
make features        # ~1 min
make train           # credit, ~17 min on 12 cores
make train-fraud     # ~4 min
make evaluate        # re-scores the frozen test sets from the saved artifacts
python scripts/benchmark_serving.py --url http://localhost:8001
```
