# EDA findings

Notebooks: `ml/notebooks/01_eda.ipynb`, `02_target_analysis.ipynb`, `03_fraud_eda.ipynb`.
All numbers below are computed on the full Kaggle training data.

## Home Credit Default Risk

| | |
|---|---|
| Applications | 307,511 × 122 columns (+ 6 history tables, ~58M rows) |
| Default rate (`TARGET=1`) | **8.07%** → imbalanced: ROC-AUC/PR-AUC/KS, class weights, calibration |
| Columns > 50% missing | 41 (mostly building-information `*_AVG/_MODE/_MEDI`) – kept; GBDTs use NaN natively, WOE gives NaN its own bin |

### 1. External bureau scores dominate
Default rate by quintile (lowest → highest score):

| | Q1 | Q2 | Q3 | Q4 | Q5 | missing |
|---|---|---|---|---|---|---|
| EXT_SOURCE_1 | 14.6% | 8.7% | 6.4% | 4.8% | 3.0% | 8.5% |
| EXT_SOURCE_2 | 15.2% | 9.2% | 7.0% | 5.4% | 3.6% | 7.9% |
| EXT_SOURCE_3 | 16.4% | 8.8% | 5.8% | 4.5% | 3.4% | 9.3% |

Combining them helps: |corr| with the target is 0.22 for `EXT_SOURCE_MEAN` vs 0.16–0.18 for each
single score → mean / min / max / product / weighted features in `features/application.py`.

### 2. `DAYS_EMPLOYED = 365243` anomaly
55,374 rows (**18.0%**) – all pensioners/unemployed. Their default rate is **5.4%** vs **8.7%**
for everyone else, so the value carries signal but would distort any ratio. Replaced by NaN
plus a `DAYS_EMPLOYED_ANOM` flag; the pandera schema allows exactly this value as the only
positive one.

### 3. Strong demographic gradients (fairness relevance)
| Segment | Default rate |
|---|---|
| Age < 25 | 12.3% |
| 25–34 | 10.7% |
| 35–44 | 8.4% |
| 45–54 | 7.1% |
| 55+ | 5.2% |
| Male | 10.1% |
| Female | 7.0% |

→ `CODE_GENDER` is **excluded from the model inputs** and both attributes are audited on the
decisions (see `fairness.md`).

### 4. Other signals
* Education: academic degree 1.8% → lower secondary 10.9%.
* Revolving loans default less (5.5%) than cash loans (8.4%).
* Income types "Unemployed" (36%) and "Maternity leave" (40%) are tiny groups (22 and 5 rows)
  → smoothed target encoding (`smoothing=20`) prevents over-fitting to them.

### 5. Leakage checks
* History tables contain only past events (`DAYS_CREDIT ≤ 0`, `DAYS_DECISION ≤ 0`).
* `SK_ID_CURR` is never a feature (used only for the ordered robustness split).
* Target encoding is **out-of-fold**; a unit test proves a row's own label cannot leak into
  its encoding.

## IEEE-CIS Fraud Detection
| | |
|---|---|
| Transactions | 590,540 over 182 days |
| Fraud rate | **3.50%** |
| By product | C 11.7% · S 5.9% · H 4.8% · R 3.8% · W 2.0% |

* The same cards recur across the period → a **random split leaks the future**; we use a time
  split (train 70% / valid 10% / test last 20% of time).
* Cards are not clients: the client id is reconstructed as `card1 + addr1 + D1n` where
  `D1n = day − D1` (the day the card was first seen) – the key idea of the winning solutions.
