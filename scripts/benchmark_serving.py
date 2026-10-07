"""Latency / throughput benchmark of the serving API with real Kaggle rows.

python scripts/benchmark_serving.py --url http://localhost:8001 --seconds 20 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

DEMO = Path(__file__).resolve().parents[1] / "backend" / "demo_data"


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def sequential(client: httpx.Client, path: str, bodies: list[dict], key: str) -> dict:
    server, total = [], []
    for b in bodies:
        t0 = time.perf_counter()
        r = client.post(path, json=b).json()
        total.append((time.perf_counter() - t0) * 1000)
        server.append(r["latency_ms"])
    return {
        "name": key,
        "n": len(bodies),
        "server_p50": statistics.median(server),
        "server_p95": pct(server, 0.95),
        "http_p50": statistics.median(total),
        "http_p95": pct(total, 0.95),
    }


async def throughput(
    url: str, path: str, bodies: list[dict], seconds: float, concurrency: int
) -> float:
    done = 0
    stop = time.perf_counter() + seconds

    async def worker(i: int) -> None:
        nonlocal done
        async with httpx.AsyncClient(base_url=url, timeout=30) as c:
            j = i
            while time.perf_counter() < stop:
                r = await c.post(path, json=bodies[j % len(bodies)])
                r.raise_for_status()
                done += 1
                j += concurrency

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(concurrency)))
    return done / (time.perf_counter() - t0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    apps = json.loads((DEMO / "applications.json").read_text())
    txs = json.loads((DEMO / "transactions.json").read_text())
    credit = [{"features": {**r["features"], **r["history"]}, "explain": True} for r in apps]
    fraud_fast = [{"transaction": t, "record_history": False} for t in txs]
    fraud_full = [{"transaction": t, "record_history": False, "explain": True} for t in txs]

    with httpx.Client(base_url=a.url, timeout=30) as c:
        for b in (credit[:5], fraud_fast[:5]):  # warm-up
            for x in b:
                c.post("/v1/credit/score" if "features" in x else "/v1/fraud/score", json=x)
        rows = [
            sequential(c, "/v1/credit/score", credit[: a.n], "credit (stacking + SHAP top-5)"),
            sequential(
                c, "/v1/fraud/score", fraud_fast[: a.n], "fraud (auth path, SHAP only if flagged)"
            ),
            sequential(c, "/v1/fraud/score", fraud_full[: min(a.n, 50)], "fraud (SHAP always)"),
        ]
    print(
        "| endpoint | n | serving p50 | serving p95 | HTTP p50 | HTTP p95 |\n|---|---|---|---|---|---|"
    )
    for r in rows:
        print(
            f"| {r['name']} | {r['n']} | {r['server_p50']:.1f} ms | {r['server_p95']:.1f} ms | "
            f"{r['http_p50']:.1f} ms | {r['http_p95']:.1f} ms |"
        )
    for name, path, bodies in (
        ("credit", "/v1/credit/score", credit),
        ("fraud", "/v1/fraud/score", fraud_fast),
    ):
        rps = asyncio.run(throughput(a.url, path, bodies, a.seconds, a.concurrency))
        print(f"throughput {name}: {rps:.0f} req/s (concurrency {a.concurrency})")


if __name__ == "__main__":
    main()
