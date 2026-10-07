"""Base settings (12-factor: everything configurable from the environment)."""

from datetime import timedelta
from pathlib import Path

import environ
from celery.schedules import crontab

BASE_DIR = Path(__file__).resolve().parents[2]
env = environ.Env()
environ.Env.read_env(BASE_DIR.parent / ".env", overwrite=False)

SECRET_KEY = env("DJANGO_SECRET_KEY", default="insecure-dev-key-change-me")
DEBUG = env.bool("DJANGO_DEBUG", default=False)
ALLOWED_HOSTS = env.list("DJANGO_ALLOWED_HOSTS", default=["localhost", "127.0.0.1", "backend"])

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # third party
    "rest_framework",
    "rest_framework_simplejwt.token_blacklist",
    "django_filters",
    "drf_spectacular",
    "corsheaders",
    "django_celery_beat",
    # local
    "apps.core",
    "apps.users",
    "apps.audit",
    "apps.applicants",
    "apps.scoring",
    "apps.transactions",
    "apps.monitoring",
    "apps.mlmodels",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "corsheaders.middleware.CorsMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "apps.core.middleware.RequestContextMiddleware",
]

ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
AUTH_USER_MODEL = "users.User"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ]
        },
    }
]

DATABASES = {
    "default": env.db(
        "DATABASE_URL", default="postgres://risklens:risklens@localhost:5432/risklens"
    )
}
DATABASES["default"]["CONN_MAX_AGE"] = env.int("DB_CONN_MAX_AGE", default=60)

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ----------------------------------------------------------------- storage
# Drift reports / uploads go to MinIO (S3) when configured, local disk otherwise.
AWS_STORAGE_BUCKET_NAME = env("S3_BUCKET", default="")
if AWS_STORAGE_BUCKET_NAME:
    STORAGES = {
        "default": {"BACKEND": "storages.backends.s3.S3Storage"},
        "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
    }
    AWS_S3_ENDPOINT_URL = env("S3_ENDPOINT_URL", default=None)
    AWS_ACCESS_KEY_ID = env("AWS_ACCESS_KEY_ID", default=None)
    AWS_SECRET_ACCESS_KEY = env("AWS_SECRET_ACCESS_KEY", default=None)
    AWS_S3_ADDRESSING_STYLE = "path"
    AWS_QUERYSTRING_AUTH = True
    AWS_DEFAULT_ACL = None

# --------------------------------------------------------------------- DRF
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": ["rest_framework_simplejwt.authentication.JWTAuthentication"],
    "DEFAULT_PERMISSION_CLASSES": ["rest_framework.permissions.IsAuthenticated"],
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.OrderingFilter",
        "rest_framework.filters.SearchFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "apps.core.pagination.StandardPagination",
    "DEFAULT_SCHEMA_CLASS": "drf_spectacular.openapi.AutoSchema",
    "EXCEPTION_HANDLER": "apps.core.exceptions.api_exception_handler",
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.UserRateThrottle",
        "rest_framework.throttling.ScopedRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "user": env("THROTTLE_USER", default="2000/hour"),
        "scoring": env("THROTTLE_SCORING", default="120/minute"),
        "fraud": env("THROTTLE_FRAUD", default="1200/minute"),
    },
}

SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=env.int("JWT_ACCESS_MINUTES", default=15)),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=env.int("JWT_REFRESH_DAYS", default=1)),
    "ROTATE_REFRESH_TOKENS": True,
    "BLACKLIST_AFTER_ROTATION": True,
    "UPDATE_LAST_LOGIN": True,
}

SPECTACULAR_SETTINGS = {
    "TITLE": "RiskLens API",
    "DESCRIPTION": "Credit-risk & fraud scoring platform: applications, decisions, audit, monitoring.",
    "VERSION": "0.1.0",
    "SERVE_INCLUDE_SCHEMA": False,
    "COMPONENT_SPLIT_REQUEST": True,
    "ENUM_NAME_OVERRIDES": {
        "ApplicationStatusEnum": "apps.applicants.models.Application.Status",
        "TransactionStatusEnum": "apps.transactions.models.Transaction.Status",
        "DriftStatusEnum": "apps.monitoring.models.DriftReport.Status",
    },
}

CORS_ALLOWED_ORIGINS = env.list(
    "CORS_ALLOWED_ORIGINS", default=["http://localhost:3000", "http://localhost:5173"]
)

# ------------------------------------------------------------------ serving
SERVING_URL = env("SERVING_URL", default="http://localhost:8001")
SERVING_TIMEOUT_SECONDS = env.float("SERVING_TIMEOUT_SECONDS", default=5.0)
SERVING_RETRIES = env.int("SERVING_RETRIES", default=3)
SERVING_INTERNAL_TOKEN = env("SERVING_INTERNAL_TOKEN", default="change-me")

MLFLOW_TRACKING_URI = env("MLFLOW_TRACKING_URI", default="http://localhost:5000")
CREDIT_MODEL_NAME = env("CREDIT_MODEL_NAME", default="risklens-credit")
FRAUD_MODEL_NAME = env("FRAUD_MODEL_NAME", default="risklens-fraud")

# ---------------------------------------------------------------- monitoring
DRIFT_PSI_THRESHOLD = env.float("DRIFT_PSI_THRESHOLD", default=0.2)
DRIFT_WINDOW_HOURS = env.int("DRIFT_WINDOW_HOURS", default=24)
DRIFT_MIN_SAMPLES = env.int("DRIFT_MIN_SAMPLES", default=500)  # PSI on < ~500 rows is mostly noise
AUTO_RETRAIN_ON_DRIFT = env.bool("AUTO_RETRAIN_ON_DRIFT", default=False)
GITHUB_REPOSITORY = env("GITHUB_REPOSITORY", default="")
GITHUB_TOKEN = env("GITHUB_TOKEN", default="")

# ------------------------------------------------------------------- celery
CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://localhost:6379/0")
CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="redis://localhost:6379/1")
CELERY_TASK_ACKS_LATE = True
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_TIME_LIMIT = 60 * 30
CELERY_BEAT_SCHEDULER = "django_celery_beat.schedulers:DatabaseScheduler"
CELERY_BEAT_SCHEDULE = {
    "nightly-drift-credit": {
        "task": "apps.monitoring.tasks.run_drift_report",
        "schedule": crontab(hour=2, minute=0),
        "args": ("credit",),
    },
    "nightly-drift-fraud": {
        "task": "apps.monitoring.tasks.run_drift_report",
        "schedule": crontab(hour=2, minute=15),
        "args": ("fraud",),
    },
    "batch-score-pending": {
        "task": "apps.scoring.tasks.batch_score_pending",
        "schedule": crontab(minute="*/15"),
    },
}

# ------------------------------------------------------------------ logging
LOG_LEVEL = env("LOG_LEVEL", default="INFO")
LOG_JSON = env.bool("LOG_JSON", default=True)
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {"console": {"class": "logging.StreamHandler"}},
    "root": {"handlers": ["console"], "level": LOG_LEVEL},
    "loggers": {"django.db.backends": {"level": "WARNING"}},
}
