"""Online history stores for fraud velocity features (implements risklens HistoryStore).

Memory store: per-process, fine for a single replica / demo.
Redis store:  shared across replicas (LPUSH + LTRIM per card, JSON events).
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict, deque
from typing import Any


class InMemoryHistoryStore:
    def __init__(self, max_events: int = 500) -> None:
        self.max_events = max_events
        self._data: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=max_events))
        self._lock = threading.Lock()

    def get(self, key: str) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._data.get(key, ()))

    def append(self, key: str, event: dict[str, Any]) -> None:
        with self._lock:
            self._data[key].append(event)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


class RedisHistoryStore:
    def __init__(self, url: str, max_events: int = 500, prefix: str = "risklens:hist:") -> None:
        import redis

        self.r = redis.Redis.from_url(url)
        self.max_events = max_events
        self.prefix = prefix

    def get(self, key: str) -> list[dict[str, Any]]:
        raw = self.r.lrange(self.prefix + key, 0, -1)
        return [json.loads(x) for x in reversed(raw)]  # oldest first

    def append(self, key: str, event: dict[str, Any]) -> None:
        k = self.prefix + key
        pipe = self.r.pipeline()
        pipe.lpush(k, json.dumps(event, default=str))
        pipe.ltrim(k, 0, self.max_events - 1)
        pipe.expire(k, 60 * 60 * 24 * 30)
        pipe.execute()
