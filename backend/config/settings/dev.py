from .base import *  # noqa: F403
from .base import env

DEBUG = env.bool("DJANGO_DEBUG", default=True)
ALLOWED_HOSTS = ["*"]
LOG_JSON = env.bool("LOG_JSON", default=False)
CORS_ALLOW_ALL_ORIGINS = True
