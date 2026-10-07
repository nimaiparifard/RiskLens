"""Tiny MLflow REST client (no heavy mlflow dependency in the backend image)."""

from __future__ import annotations

from typing import Any

import httpx
from django.conf import settings

from apps.core.exceptions import ExternalServiceError

METRIC_KEYS = (
    "roc_auc",
    "pr_auc",
    "ks",
    "gini",
    "brier",
    "expected_cost",
    "loss_saved_pct",
    "approval_rate",
    "recall_at_1pct_fpr",
    "fairness_max_dp_gap",
    "oot_roc_auc",
    "threshold",
)


class MlflowRegistryClient:
    def __init__(
        self, base_url: str | None = None, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.base_url = (base_url or settings.MLFLOW_TRACKING_URI).rstrip("/")
        self._transport = transport

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        try:
            with httpx.Client(base_url=self.base_url, timeout=5, transport=self._transport) as c:
                r = c.get(f"/api/2.0/mlflow/{path}", params=params)
                r.raise_for_status()
                return r.json()
        except httpx.HTTPError as exc:
            raise ExternalServiceError(f"MLflow unavailable: {exc}") from exc

    def versions(self, name: str) -> list[dict[str, Any]]:
        data = self._get("model-versions/search", filter=f"name='{name}'", max_results=50)
        return data.get("model_versions", [])

    def aliases(self, name: str) -> dict[str, str]:
        model = self._get("registered-models/get", name=name).get("registered_model", {})
        return {a["alias"]: a["version"] for a in model.get("aliases", [])}

    def run_metrics(self, run_id: str) -> dict[str, float]:
        run = self._get("runs/get", run_id=run_id).get("run", {})
        metrics = {m["key"]: m["value"] for m in run.get("data", {}).get("metrics", [])}
        return {k: metrics[k] for k in METRIC_KEYS if k in metrics}

    def model_summary(self, name: str) -> dict[str, Any]:
        aliases = self.aliases(name)
        by_version = {v: a for a, v in aliases.items()}
        versions = []
        for v in sorted(self.versions(name), key=lambda x: int(x["version"]), reverse=True)[:10]:
            versions.append(
                {
                    "version": v["version"],
                    "aliases": [a for a, ver in aliases.items() if ver == v["version"]],
                    "created_at": v.get("creation_timestamp"),
                    "run_id": v.get("run_id"),
                    "status": v.get("status"),
                    "metrics": self.run_metrics(v["run_id"]) if v.get("run_id") else {},
                    "is_champion": by_version.get(v["version"]) == "champion",
                }
            )
        return {"name": name, "aliases": aliases, "versions": versions}
