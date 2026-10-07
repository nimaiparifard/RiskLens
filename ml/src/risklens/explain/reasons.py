"""Turn SHAP values into top-k human-readable reason codes."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from risklens.interfaces import Reason

DEFAULT_PATH = Path(__file__).with_name("reason_codes.yaml")
_ENCODED = re.compile(r"(_TE|_FREQ|=.*)$")


def base_feature(name: str) -> str:
    """ORGANIZATION_TYPE_TE -> ORGANIZATION_TYPE; FLAG_OWN_CAR=Y -> FLAG_OWN_CAR."""
    return _ENCODED.sub("", name)


def humanize(name: str) -> str:
    return name.replace("_", " ").lower()


class ReasonCoder:
    def __init__(self, mapping: dict[str, Any] | None = None) -> None:
        if mapping is None:
            mapping = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
        self.features: dict[str, dict[str, str]] = mapping.get("features", {})
        self.prefixes: dict[str, str] = mapping.get("prefixes", {})
        # group -> list of feature names / prefixes that express the same underlying fact
        self.groups: dict[str, list[str]] = mapping.get("groups", {})

    def group_of(self, feature: str) -> str:
        """Related features (EXT_SOURCE_1/2/3/MEAN/MIN/...) collapse into one reason."""
        base = base_feature(feature)
        if "groups" not in self.__dict__:  # scorers pickled before groups existed
            default = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
            self.groups = default.get("groups", {})
        for group, members in self.groups.items():
            if any(base == m or (m.endswith("*") and base.startswith(m[:-1])) for m in members):
                return group
        return base

    def text(self, feature: str, direction: str) -> str:
        base = base_feature(feature)
        for key in (feature, base):
            if key in self.features:
                return self.features[key].get(direction, humanize(key))
        verb = "raises" if direction == "increases_risk" else "lowers"
        for prefix, label in self.prefixes.items():
            if base.startswith(prefix):
                return f"{label.capitalize()} ({humanize(base)}) {verb} risk"
        return f"{humanize(base).capitalize()} {verb} risk"

    def top_reasons(
        self,
        contributions: np.ndarray,
        feature_names: list[str],
        raw_row: pd.Series | dict[str, Any] | None,
        k: int = 5,
        only_adverse: bool = False,
    ) -> list[Reason]:
        """One reason per feature group, ranked by the group's total |SHAP|.

        SHAP values are additive, so summing within a group is exact. The text and
        value come from the group member that pushes hardest in the group's direction.
        With ``only_adverse`` only risk-increasing groups are returned.
        """
        totals: dict[str, float] = {}
        members: dict[str, list[int]] = {}
        for j, name in enumerate(feature_names):
            c = float(contributions[j])
            if c == 0.0:
                continue
            g = self.group_of(name)
            totals[g] = totals.get(g, 0.0) + c
            members.setdefault(g, []).append(j)

        out: list[Reason] = []
        for g in sorted(totals, key=lambda key: -abs(totals[key])):
            total = totals[g]
            if total == 0.0 or (only_adverse and total <= 0):
                continue
            sign = 1.0 if total > 0 else -1.0
            rep = max(members[g], key=lambda j: sign * float(contributions[j]))
            feature = feature_names[rep]
            base = base_feature(feature)
            direction = "increases_risk" if total > 0 else "decreases_risk"
            value = None
            if raw_row is not None:
                value = raw_row.get(base, raw_row.get(feature))
            value = _jsonable(value)
            text = self.text(feature, direction)
            if raw_row is not None and value is None and base == feature:
                # the effect comes from the input being ABSENT - say so instead of
                # describing a value the applicant never gave us
                verb = "raises" if direction == "increases_risk" else "lowers"
                text = f"Information not provided ({humanize(base)}) {verb} risk"
            out.append(
                Reason(
                    feature=base,
                    value=value,
                    contribution=round(total, 4),
                    direction=direction,
                    text=text,
                )
            )
            if len(out) >= k:
                break
        return out


def _jsonable(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if math.isnan(f) else round(f, 6)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v
