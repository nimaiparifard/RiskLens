from django.apps import AppConfig


class CoreConfig(AppConfig):
    name = "apps.core"

    def ready(self) -> None:
        from django.conf import settings

        from apps.core.logging import configure_structlog

        configure_structlog(settings.LOG_LEVEL, settings.LOG_JSON)
