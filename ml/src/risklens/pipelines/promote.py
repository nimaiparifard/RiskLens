"""Champion / challenger promotion.

Rule (configs/*.yaml -> promotion):
    candidate.primary >= champion.primary + min_improvement
    AND every guard metric is not worse (e.g. expected_cost <= champion's)
    AND fairness gap <= max_fairness_gap
  -> alias "challenger"
A human then approves (GitHub "production" environment) -> alias "champion".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from risklens.config import PromotionConfig

log = logging.getLogger(__name__)

CHAMPION, CHALLENGER = "champion", "challenger"


@dataclass
class PromotionDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)
    candidate: dict[str, float] = field(default_factory=dict)
    champion: dict[str, float] | None = None

    def to_markdown(self) -> str:
        status = "PROMOTE to challenger" if self.promote else "KEEP current champion"
        lines = [
            f"### Promotion decision: **{status}**",
            "",
            "| metric | candidate | champion |",
            "|---|---|---|",
        ]
        keys = sorted(set(self.candidate) | set(self.champion or {}))
        for k in keys:
            if any(
                s in k
                for s in (
                    "roc_auc",
                    "pr_auc",
                    "ks",
                    "brier",
                    "expected_cost",
                    "recall",
                    "fairness_max",
                )
            ):
                c = self.candidate.get(k)
                ch = (self.champion or {}).get(k)
                lines.append(f"| {k} | {_fmt(c)} | {_fmt(ch)} |")
        lines += ["", *[f"- {r}" for r in self.reasons]]
        return "\n".join(lines)


def _fmt(v: float | None) -> str:
    return "—" if v is None else f"{v:,.4f}"


def should_promote(
    candidate: dict[str, float], champion: dict[str, float] | None, rules: PromotionConfig
) -> PromotionDecision:
    """Pure decision function (unit-tested, no MLflow)."""
    reasons: list[str] = []
    ok = True
    fair = candidate.get("fairness_max_dp_gap")
    if fair is not None and fair > rules.max_fairness_gap:
        ok = False
        reasons.append(f"fairness gap {fair:.4f} > {rules.max_fairness_gap}")
    if champion is None:
        reasons.append("no champion yet -> candidate becomes challenger")
        return PromotionDecision(ok, reasons, candidate, None)

    pm = rules.primary_metric
    c_val, ch_val = candidate.get(pm), champion.get(pm)
    if c_val is None or ch_val is None:
        return PromotionDecision(False, [f"primary metric '{pm}' missing"], candidate, champion)
    if c_val >= ch_val + rules.min_improvement:
        reasons.append(f"{pm} {c_val:.4f} >= {ch_val:.4f} + {rules.min_improvement}")
    else:
        ok = False
        reasons.append(f"{pm} {c_val:.4f} < {ch_val:.4f} + {rules.min_improvement}")
    for metric, direction in rules.guard_metrics.items():
        c, ch = candidate.get(metric), champion.get(metric)
        if c is None or ch is None:
            continue
        worse = c > ch if direction == "lower" else c < ch
        if worse:
            ok = False
            reasons.append(f"guard '{metric}' worse: {c:,.4f} vs {ch:,.4f}")
        else:
            reasons.append(f"guard '{metric}' ok: {c:,.4f} vs {ch:,.4f}")
    return PromotionDecision(ok, reasons, candidate, champion)


# ----------------------------------------------------------- MLflow side
def _client() -> Any:
    from mlflow import MlflowClient

    return MlflowClient()


def version_metrics(model_name: str, version: str) -> dict[str, float]:
    client = _client()
    mv = client.get_model_version(model_name, version)
    if not mv.run_id:
        return {}
    return dict(client.get_run(mv.run_id).data.metrics)


def alias_version(model_name: str, alias: str) -> str | None:
    try:
        return _client().get_model_version_by_alias(model_name, alias).version
    except Exception:
        return None


def latest_version(model_name: str) -> str | None:
    versions = _client().search_model_versions(f"name='{model_name}'")
    return max((v.version for v in versions), key=int, default=None)


def evaluate_candidate(
    model_name: str, rules: PromotionConfig, version: str | None = None
) -> PromotionDecision:
    """Compare a registered version with the champion; set 'challenger' if it wins."""
    client = _client()
    version = version or latest_version(model_name)
    if version is None:
        raise RuntimeError(f"no versions registered for {model_name}")
    cand = version_metrics(model_name, version)
    champ_v = alias_version(model_name, CHAMPION)
    champ = version_metrics(model_name, champ_v) if champ_v else None
    decision = should_promote(cand, champ, rules)
    if decision.promote:
        client.set_registered_model_alias(model_name, CHALLENGER, version)
        log.info("version %s -> %s", version, CHALLENGER)
    client.set_model_version_tag(
        model_name, version, "promotion_decision", "challenger" if decision.promote else "rejected"
    )
    return decision


def approve_challenger(model_name: str) -> str:
    """Human-approved step: challenger -> champion (previous champion tagged)."""
    client = _client()
    chall = alias_version(model_name, CHALLENGER)
    if chall is None:
        raise RuntimeError("no challenger to approve")
    prev = alias_version(model_name, CHAMPION)
    if prev:
        client.set_model_version_tag(model_name, prev, "retired_by", chall)
    client.set_registered_model_alias(model_name, CHAMPION, chall)
    client.delete_registered_model_alias(model_name, CHALLENGER)
    log.info("version %s is now champion (was %s)", chall, prev)
    return chall


def registry_summary(model_name: str) -> dict[str, Any]:
    out: dict[str, Any] = {"name": model_name}
    for alias in (CHAMPION, CHALLENGER):
        v = alias_version(model_name, alias)
        out[alias] = {"version": v, "metrics": version_metrics(model_name, v) if v else {}}
    return out
