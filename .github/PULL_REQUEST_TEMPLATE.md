## What & why

<!-- one paragraph: the problem and the change -->

## Type
- [ ] feature  - [ ] fix  - [ ] ML (features / model / config)  - [ ] infra / CI  - [ ] docs

## ML changes (if any)
- [ ] `ml-eval` quality gate passes (AUC drop ≤ 0.01 on the smoke set)
- [ ] same feature function used in training and serving (no skew)
- [ ] no new leakage (target encoding is out-of-fold, fraud split is time-based)
- [ ] config / thresholds changed in `ml/configs/`, not in code

## Checklist
- [ ] tests added / updated (`make test`)
- [ ] `make lint` clean
- [ ] migrations included (`makemigrations --check`)
- [ ] API change → `make types` regenerated, docs/api.md updated
- [ ] no secrets, `.env.example` updated for new variables
