# Build context: repository root (the worker imports the ml/ package for Evidently reports)
# ---------- builder
FROM python:3.15-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install uv
WORKDIR /build
COPY ml/ ml/
COPY backend/pyproject.toml backend/pyproject.toml
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install "./ml[monitoring]" -r backend/pyproject.toml

# ---------- runtime
FROM python:3.15-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    DJANGO_SETTINGS_MODULE=config.settings.prod
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app backend/ /app/
RUN DJANGO_SECRET_KEY=build python manage.py collectstatic --noinput
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --retries=5 CMD curl -fs http://localhost:8000/api/health/ || exit 1
CMD ["sh", "-c", "python manage.py migrate --noinput && gunicorn config.wsgi:application -b 0.0.0.0:8000 -w ${WEB_CONCURRENCY:-3} --access-logfile -"]
