from .base import *  # noqa: F403
from .base import REST_FRAMEWORK, env

DEBUG = False
SECRET_KEY = "test-secret"
# CI uses the Postgres service (DATABASE_URL); locally fall back to SQLite
DATABASES = {"default": env.db("DATABASE_URL", default="sqlite:///:memory:")}
PASSWORD_HASHERS = ["django.contrib.auth.hashers.MD5PasswordHasher"]
CELERY_TASK_ALWAYS_EAGER = True
CELERY_TASK_EAGER_PROPAGATES = True
CELERY_BROKER_URL = "memory://"
CELERY_RESULT_BACKEND = "cache+memory://"
STORAGES = {
    "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"},
    "staticfiles": {"BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage"},
}
SERVING_URL = "http://serving.test"
SERVING_RETRIES = 2
MLFLOW_TRACKING_URI = "http://mlflow.test"
LOG_JSON = False
REST_FRAMEWORK = {
    **REST_FRAMEWORK,
    "DEFAULT_THROTTLE_RATES": {
        "user": "100000/hour",
        "scoring": "100000/minute",
        "fraud": "100000/minute",
    },
}
