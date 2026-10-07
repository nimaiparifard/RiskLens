from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from risklens.config import CostConfig, PromotionConfig
from risklens.evaluation.cost import (
    approval_rate_at_risk,
    assign_grades,
    baseline_cost,
    expected_cost,
    optimal_threshold,
)
from risklens.evaluation.metrics import classification_metrics, ks_statistic, recall_at_fpr
from risklens.explain.reasons import ReasonCoder, base_feature
from risklens.fairness.report import fairness_report
from risklens.monitoring.drift import compute_drift, psi_categorical, psi_numeric
from risklens.pipelines.promote import should_promote


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    rng = np.random.default_rng(0)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.05


def test_recall_at_fpr():
    y = np.r_[np.zeros(100), np.ones(10)]
    p = np.r_[np.linspace(0, 0.5, 100), np.full(10, 0.9)]
    assert recall_at_fpr(y, p, 0.01) == pytest.approx(1.0)


def test_classification_metrics_keys():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    m = classification_metrics(y, np.clip(y * 0.6 + rng.random(200) * 0.4, 0, 1))
    assert m["gini"] == pytest.approx(2 * m["roc_auc"] - 1)
    assert {"pr_auc", "brier", "ks", "recall_at_1pct_fpr"} <= set(m)


def test_expected_cost_matrix():
    cfg = CostConfig(fn_cost_rate=0.5, fp_cost_rate=0.1, fp_cost_fixed=1.0)
    y = np.array([1, 0, 1, 0])
    p = np.array([0.9, 0.9, 0.1, 0.1])
    amounts = np.array([100.0, 100.0, 100.0, 100.0])
    # flag rows 0,1 -> FP row1 (100*0.1+1); FN row2 (100*0.5)
    assert expected_cost(y, p, 0.5, amounts, cfg) == pytest.approx(11 + 50)
    assert baseline_cost(y, amounts, cfg) == pytest.approx(100.0)


def test_optimal_threshold_beats_baseline():
    rng = np.random.default_rng(1)
    y = rng.random(3000) < 0.1
    p = np.clip(y * 0.3 + rng.random(3000) * 0.5, 0, 1)
    cfg = CostConfig(fn_cost_rate=0.45, fp_cost_rate=0.08)
    amounts = np.full(3000, 1000.0)
    thr, cost = optimal_threshold(y, p, amounts, cfg)
    assert cost < baseline_cost(y, amounts, cfg)
    assert 0 < thr < 1


def test_grades_monotonic():
    bands = {"A": 0.03, "B": 0.06, "C": 0.12, "D": 0.25, "E": 1.0}
    assert assign_grades(np.array([0.01, 0.03, 0.05, 0.2, 0.9]), bands).tolist() == [
        "A",
        "A",
        "B",
        "D",
        "E",
    ]


def test_approval_rate_at_risk():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    assert approval_rate_at_risk(y, p, 0.0) == 0.6


# --------------------------------------------------------------- promotion
RULES = PromotionConfig(
    min_improvement=0.002, guard_metrics={"expected_cost": "lower"}, max_fairness_gap=0.1
)


def test_promote_when_better():
    d = should_promote(
        {"roc_auc": 0.785, "expected_cost": 90}, {"roc_auc": 0.78, "expected_cost": 100}, RULES
    )
    assert d.promote


def test_no_promote_small_gain():
    d = should_promote(
        {"roc_auc": 0.781, "expected_cost": 90}, {"roc_auc": 0.78, "expected_cost": 100}, RULES
    )
    assert not d.promote


def test_no_promote_when_guard_worse():
    d = should_promote(
        {"roc_auc": 0.80, "expected_cost": 101}, {"roc_auc": 0.78, "expected_cost": 100}, RULES
    )
    assert not d.promote and any("guard" in r for r in d.reasons)


def test_no_promote_unfair():
    d = should_promote({"roc_auc": 0.9, "fairness_max_dp_gap": 0.3}, None, RULES)
    assert not d.promote


def test_first_model_becomes_challenger():
    d = should_promote({"roc_auc": 0.7}, None, RULES)
    assert d.promote and "markdown" not in d.to_markdown()


# ------------------------------------------------------------------- drift
def test_psi_stable_vs_shifted():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=5000)
    assert psi_numeric(ref, rng.normal(size=5000)) < 0.02
    assert psi_numeric(ref, rng.normal(1.0, 1, size=5000)) > 0.2


def test_psi_categorical():
    a = pd.Series(["x"] * 50 + ["y"] * 50)
    assert psi_categorical(a, a) == pytest.approx(0, abs=1e-9)
    assert psi_categorical(a, pd.Series(["x"] * 95 + ["y"] * 5)) > 0.2


def test_compute_drift_report():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame(
        {"a": rng.normal(size=2000), "b": rng.choice(["u", "v"], 2000), "score": rng.random(2000)}
    )
    cur = pd.DataFrame(
        {"a": rng.normal(2, 1, 500), "b": rng.choice(["u", "v"], 500), "score": rng.random(500)}
    )
    res = compute_drift(ref, cur)
    assert res.drifted_features == ["a"]
    assert res.drift_detected and res.prediction_psi < 0.1
    assert res.to_dict()["max_psi"] == res.max_psi


# --------------------------------------------------------------- reasons / fairness
def test_reason_codes():
    coder = ReasonCoder()
    assert base_feature("ORGANIZATION_TYPE_TE") == "ORGANIZATION_TYPE"
    assert base_feature("FLAG_OWN_CAR=Y") == "FLAG_OWN_CAR"
    reasons = coder.top_reasons(
        np.array([0.8, -0.1, 0.3, 0.05]),
        ["EXT_SOURCE_2", "AMT_CREDIT", "ORGANIZATION_TYPE_TE", "ORGANIZATION_TYPE_FREQ"],
        {"EXT_SOURCE_2": 0.1, "ORGANIZATION_TYPE": "Bank"},
        k=5,
    )
    assert reasons[0].text == "Low external credit bureau score (bureau 2)"
    assert [r.feature for r in reasons] == [
        "EXT_SOURCE_2",
        "ORGANIZATION_TYPE",
        "AMT_CREDIT",
    ]  # deduped
    assert reasons[1].value == "Bank"
    assert "lowers risk" in coder.text("BURO_SOMETHING_NEW", "decreases_risk")


def test_reason_for_missing_input_says_not_provided():
    coder = ReasonCoder()
    r = coder.top_reasons(
        np.array([0.4]), ["AMT_GOODS_PRICE"], {"AMT_GOODS_PRICE": float("nan")}, k=1
    )
    assert r[0].value is None
    assert r[0].text == "Information not provided (amt goods price) raises risk"
    r = coder.top_reasons(np.array([0.4]), ["AMT_GOODS_PRICE"], {"AMT_GOODS_PRICE": 1000.0}, k=1)
    assert "not provided" not in r[0].text


def test_reason_groups_collapse_related_features():
    coder = ReasonCoder()
    names = [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_MIN",
        "EXT_SOURCE_3",
        "AGE_YEARS",
        "DAYS_BIRTH",
        "AMT_CREDIT",
    ]
    contrib = np.array([0.5, 0.3, -0.1, 0.2, 0.15, -0.4])
    reasons = coder.top_reasons(contrib, names, {"EXT_SOURCE_MEAN": 0.2}, k=5)
    # external scores: 0.5 + 0.3 - 0.1 = 0.7 ; age: 0.35 ; AMT_CREDIT: -0.4
    assert [r.feature for r in reasons] == ["EXT_SOURCE_MEAN", "AMT_CREDIT", "AGE_YEARS"]
    assert reasons[0].contribution == pytest.approx(0.7)
    assert reasons[2].contribution == pytest.approx(0.35)
    assert len({coder.group_of(n) for n in names[:3]}) == 1
    # SHAP additivity is preserved: group totals sum to the original total
    assert sum(r.contribution for r in reasons) == pytest.approx(contrib.sum())


def test_fairness_report():
    y = np.array([0, 1, 0, 1, 0, 0, 1, 0] * 10)
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.7, 0.1, 0.9, 0.6] * 10)
    groups = pd.DataFrame({"G": ["a", "a", "a", "a", "b", "b", "b", "b"] * 10})
    res = fairness_report(y, p, 0.5, groups, min_group_size=1)
    assert res.gaps["G"]["demographic_parity_diff"] == pytest.approx(0.25)
    assert "fairness_max_dp_gap" in res.metrics()
    assert "## G" in res.to_markdown()


def test_fairness_mitigation_threshold_optimizer():
    pytest.importorskip("fairlearn")
    rng = np.random.default_rng(0)
    n = 4000
    group = rng.choice(["young", "old"], n)
    y = (rng.random(n) < np.where(group == "young", 0.15, 0.05)).astype(int)
    p = np.clip(0.1 * y + np.where(group == "young", 0.12, 0.04) + rng.normal(0, 0.03, n), 0, 1)
    res = fairness_report(y, p, 0.12, pd.DataFrame({"AGE": group}), mitigate=True)
    m = res.mitigation["AGE"]
    assert "status" not in m, m
    assert m["approval_rate_gap_after"] < m["approval_rate_gap_before"]
    assert "Mitigation" in res.to_markdown()


def test_fairness_gate_features_only_count_gated_attributes():
    y = np.array([0, 1] * 40)
    p = np.array([0.9, 0.9, 0.1, 0.1] * 20)
    sens = pd.DataFrame({"AGE": ["young", "young", "old", "old"] * 20, "SEX": ["F", "M"] * 40})
    res = fairness_report(y, p, 0.5, sens, min_group_size=1, gate_features=["SEX"])
    assert res.gaps["AGE"]["demographic_parity_diff"] == pytest.approx(1.0)
    assert res.max_gap == pytest.approx(res.gaps["SEX"]["demographic_parity_diff"])
    assert "reported only" in res.to_markdown()
