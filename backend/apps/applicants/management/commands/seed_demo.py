"""Create demo users + applications.

Applications come from ``demo_data/applications.json`` (real, unlabelled Home
Credit test applications exported by ``scripts/export_demo_data.py``) and fall
back to randomly generated ones.

    python manage.py seed_demo --applications 60 --score
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from apps.applicants.models import Application
from apps.applicants.services import create_application
from apps.core.exceptions import DomainError
from apps.users.models import User

DEMO_FILE = Path(settings.BASE_DIR) / "demo_data" / "applications.json"
FIRST = [
    "Anna",
    "Ben",
    "Chloe",
    "David",
    "Elena",
    "Farid",
    "Grace",
    "Hugo",
    "Iris",
    "Jonas",
    "Karim",
    "Lena",
    "Mina",
    "Noah",
    "Olga",
    "Pavel",
    "Rosa",
    "Sami",
    "Tara",
    "Viktor",
    "Yara",
    "Zane",
]
LAST = [
    "Novak",
    "Schmidt",
    "Rossi",
    "Dubois",
    "Kowalski",
    "Jansen",
    "Petrov",
    "Haddad",
    "Lindqvist",
    "Moreau",
    "Ibrahim",
    "Weber",
    "Costa",
    "Nagy",
    "Fischer",
]


def random_features(rng: random.Random) -> dict:
    income = round(rng.lognormvariate(11.9, 0.5), -3)
    credit = round(income * rng.uniform(1, 5), -3)
    return {
        "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Cash loans", "Revolving loans"]),
        "CODE_GENDER": rng.choice(["F", "M"]),
        "FLAG_OWN_CAR": rng.choice(["Y", "N"]),
        "FLAG_OWN_REALTY": rng.choice(["Y", "N"]),
        "CNT_CHILDREN": rng.choice([0, 0, 1, 2]),
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": round(credit / rng.uniform(10, 30), 1),
        "AMT_GOODS_PRICE": round(credit * 0.9, -3),
        "DAYS_BIRTH": -rng.randint(22 * 365, 65 * 365),
        "DAYS_EMPLOYED": -rng.randint(100, 12 * 365),
        "NAME_EDUCATION_TYPE": rng.choice(["Secondary / secondary special", "Higher education"]),
        "NAME_INCOME_TYPE": rng.choice(["Working", "Commercial associate", "State servant"]),
        "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"]),
        "EXT_SOURCE_2": round(rng.uniform(0.05, 0.85), 4),
        "EXT_SOURCE_3": round(rng.uniform(0.05, 0.85), 4) if rng.random() > 0.2 else None,
    }


class Command(BaseCommand):
    help = "Seed demo users and loan applications"

    def add_arguments(self, parser):
        parser.add_argument("--applications", type=int, default=60)
        parser.add_argument("--score", action="store_true", help="score them via the serving API")
        parser.add_argument("--seed", type=int, default=7)

    def handle(self, *args, **opts):
        password = os.getenv("DEMO_PASSWORD", "risklens-demo")
        for username, role, staff in (
            ("admin", "admin", True),
            ("officer", "officer", False),
            ("viewer", "viewer", False),
        ):
            user, created = User.objects.get_or_create(
                username=username,
                defaults={
                    "role": role,
                    "is_staff": staff,
                    "is_superuser": staff,
                    "email": f"{username}@risklens.local",
                },
            )
            if created:
                user.set_password(password)
                user.save()
        self.stdout.write(
            f"users: admin / officer / viewer (password from DEMO_PASSWORD, default '{password}')"
        )

        if Application.objects.exists():
            self.stdout.write("applications already present – skipping")
            return
        rng = random.Random(opts["seed"])
        rows = json.loads(DEMO_FILE.read_text()) if DEMO_FILE.exists() else []
        officer = User.objects.get(username="officer")
        created = []
        for i in range(opts["applications"]):
            if rows:
                row = rows[i % len(rows)]
                features, history, ext_id = (
                    row["features"],
                    row.get("history", {}),
                    str(row.get("SK_ID_CURR", "")),
                )
            else:
                features, history, ext_id = random_features(rng), {}, ""
            name = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
            created.append(
                create_application(
                    applicant={
                        "full_name": name,
                        "external_id": ext_id,
                        "email": f"{name.lower().replace(' ', '.')}@example.com",
                    },
                    features=features,
                    history=history,
                    user=officer,
                )
            )
        self.stdout.write(
            self.style.SUCCESS(
                f"created {len(created)} applications "
                f"({'real Home Credit test rows' if rows else 'synthetic'})"
            )
        )
        if opts["score"]:
            from apps.scoring.services import score_application

            ok = 0
            for app in created[: int(len(created) * 0.7)]:  # leave some pending
                try:
                    score_application(app, officer)
                    ok += 1
                except DomainError as exc:
                    self.stderr.write(f"scoring failed: {exc.message}")
                    break
            self.stdout.write(self.style.SUCCESS(f"scored {ok} applications"))
