"""HTTP client for the internal serving API (timeouts, retries, error mapping).

Only connection errors / 5xx are retried (idempotent scoring); 4xx are the
caller's fault and surface immediately as ``ServingRejected``.
"""

from __future__ import annotations

from typing import Any

import httpx
import structlog
from django.conf import settings
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from apps.core.exceptions import ServingRejected, ServingUnavailable
from apps.core.request_context import get_request_id

log = structlog.get_logger(__name__)


def _retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500


class ServingClient:
    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        retries: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = (base_url or settings.SERVING_URL).rstrip("/")
        self.timeout = httpx.Timeout(timeout or settings.SERVING_TIMEOUT_SECONDS, connect=2.0)
        self.retries = retries or settings.SERVING_RETRIES
        self._transport = transport

    def _client(self) -> httpx.Client:
        headers = {"X-Internal-Token": settings.SERVING_INTERNAL_TOKEN}
        if rid := get_request_id():
            headers["X-Request-ID"] = rid
        return httpx.Client(
            base_url=self.base_url, timeout=self.timeout, headers=headers, transport=self._transport
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        @retry(
            retry=retry_if_exception(_retryable),
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=0.2, max=2),
            reraise=True,
        )
        def _do() -> Any:
            with self._client() as c:
                r = c.request(method, path, **kwargs)
                r.raise_for_status()
                return r.json()

        try:
            return _do()
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            try:
                detail = exc.response.json()
            except ValueError:
                detail = exc.response.text
            if status == 503 or status >= 500:
                raise ServingUnavailable(
                    "model server unavailable", upstream_status=status, upstream=detail
                ) from exc
            raise ServingRejected(
                "model server rejected the request", upstream_status=status, upstream=detail
            ) from exc
        except httpx.TransportError as exc:
            log.error("serving_unreachable", url=self.base_url, error=str(exc))
            raise ServingUnavailable("model server unreachable") from exc

    # ---------------------------------------------------------------- API
    def score_credit(self, features: dict[str, Any], explain: bool = True) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/credit/score", json={"features": features, "explain": explain}
        )

    def score_credit_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # explain=True: every stored score must carry its reason codes (compliance)
        return self._request(
            "POST", "/v1/credit/score/batch", json={"items": items, "explain": True}
        )["results"]

    def score_fraud(
        self, transaction: dict[str, Any], explain: bool | str = "flagged"
    ) -> dict[str, Any]:
        return self._request(
            "POST", "/v1/fraud/score", json={"transaction": transaction, "explain": explain}
        )

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/v1/model/info")

    def reload(self, task: str | None = None) -> dict[str, Any]:
        return self._request("POST", "/v1/model/reload", params={"task": task} if task else None)

    def reference(self, task: str, n: int = 2000) -> dict[str, Any]:
        return self._request("GET", "/v1/model/reference", params={"task": task, "n": n})

    def drift(
        self, task: str, records: list[dict[str, Any]], scores: list[float], psi_threshold: float
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/monitoring/drift",
            json={
                "task": task,
                "records": records,
                "scores": scores,
                "psi_threshold": psi_threshold,
            },
        )


def get_serving_client() -> ServingClient:
    """Indirection so tests can monkeypatch a fake client."""
    return ServingClient()
