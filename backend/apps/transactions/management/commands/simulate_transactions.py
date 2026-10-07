"""Replay transactions as a live stream for the fraud monitor demo.

Uses ``demo_data/transactions.json`` (real IEEE-CIS rows exported by
``scripts/export_demo_data.py``) or generates random ones.

    python manage.py simulate_transactions --rate 5 --count 500
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.core.exceptions import DomainError
from apps.transactions.models import Transaction
from apps.transactions.services import score_transaction

DEMO_FILE = Path(settings.BASE_DIR) / "demo_data" / "transactions.json"


def random_tx(rng: random.Random, i: int, t: int) -> dict:
    card = rng.choice([9500, 15885, 17188, 12695, 2803, 7919])
    fraudy = rng.random() < 0.05
    return {
        "TransactionID": 9_000_000 + i,
        "TransactionDT": t,
        "TransactionAmt": round(rng.lognormvariate(4, 1) * (5 if fraudy else 1), 2),
        "ProductCD": rng.choice(["W", "W", "W", "C", "H", "R"]),
        "card1": card,
        "card4": rng.choice(["visa", "mastercard"]),
        "card6": rng.choice(["debit", "credit"]),
        "addr1": rng.choice([204.0, 299.0, 325.0]),
        "D1": float(rng.randint(0, 400)),
        "P_emaildomain": rng.choice(
            ["gmail.com", "yahoo.com", "hotmail.com", "protonmail.com" if fraudy else "gmail.com"]
        ),
        "DeviceType": rng.choice(["desktop", "mobile", None]),
    }


class Command(BaseCommand):
    help = "Stream transactions through the fraud scoring API"

    def add_arguments(self, parser):
        parser.add_argument("--rate", type=float, default=2.0, help="transactions per second")
        parser.add_argument("--count", type=int, default=200)
        parser.add_argument("--seed", type=int, default=1)

    def handle(self, *args, **opts):
        rng = random.Random(opts["seed"])
        rows = json.loads(DEMO_FILE.read_text()) if DEMO_FILE.exists() else []
        offset = (
            Transaction.objects.count() // max(len(rows), 1)
        ) * 10_000_000  # unique ids across replays
        t = 15_000_000
        for i in range(opts["count"]):
            if rows:
                tx = dict(rows[(Transaction.objects.count() + i) % len(rows)])
                tx["TransactionID"] = int(tx["TransactionID"]) + offset
            else:
                t += rng.randint(5, 600)
                tx = random_tx(rng, Transaction.objects.count() + i, t)
            try:
                res = score_transaction(tx)
                flag = "FLAGGED" if res.is_flagged else "ok"
                self.stdout.write(
                    f"{tx['TransactionID']} amt={tx['TransactionAmt']:>9} p={res.probability:.4f} {res.decision:8} {flag}"
                )
            except DomainError as exc:
                self.stderr.write(f"error: {exc.message}")
                time.sleep(2)
            time.sleep(1.0 / opts["rate"])
