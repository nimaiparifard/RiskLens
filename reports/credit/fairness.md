# Fairness report

Decisions audited at the cost-optimal threshold. Protected attributes are
**excluded from the model's inputs**; this audit checks for indirect (proxy) bias.

## CODE_GENDER

| group | n | default_rate | approval_rate | tpr | fpr |
|---|---|---|---|---|---|
| F | 30390 | 0.0697 | 0.8817 | 0.4361 | 0.0945 |
| M | 15735 | 0.1020 | 0.8252 | 0.5196 | 0.1356 |
| missing | 2 | 0.0000 | 0.5000 |  | 0.5000 |

- Demographic parity difference (approval rate): **0.0565**
- Equalized odds difference: **0.0836**
- Promotion gate: **yes**

## AGE_GROUP

| group | n | default_rate | approval_rate | tpr | fpr |
|---|---|---|---|---|---|
| 25-34 | 10767 | 0.1062 | 0.7838 | 0.5748 | 0.1736 |
| 35-44 | 12609 | 0.0804 | 0.8581 | 0.4793 | 0.1124 |
| 45-54 | 10577 | 0.0729 | 0.8909 | 0.4384 | 0.0832 |
| 55+ | 10422 | 0.0548 | 0.9470 | 0.2434 | 0.0419 |
| <25 | 1752 | 0.1284 | 0.7015 | 0.6133 | 0.2521 |

- Demographic parity difference (approval rate): **0.2456**
- Equalized odds difference: **0.3699**
- Promotion gate: **no (reported only)**

## Mitigation: Fairlearn ThresholdOptimizer (demographic parity)

Group-specific thresholds chosen to equalise approval rates - and what that costs.

- **CODE_GENDER**: approval_rate_gap_before = 0.0565, approval_rate_gap_after = 0.0003, balanced_accuracy_before = 0.6819, balanced_accuracy_after = 0.7165, approval_rate_after = 0.7152
- **AGE_GROUP**: approval_rate_gap_before = 0.2456, approval_rate_gap_after = 0.0019, balanced_accuracy_before = 0.6819, balanced_accuracy_after = 0.7144, approval_rate_after = 0.6762