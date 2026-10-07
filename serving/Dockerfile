# Build context: repository root (installs the shared ml/ package = same feature code as training)
FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN pip install uv
WORKDIR /build
COPY ml/ ml/
COPY serving/pyproject.toml serving/pyproject.toml
# one resolution: the local ./ml satisfies serving's `risklens[serve]` requirement
RUN uv venv /opt/venv && \
    VIRTUAL_ENV=/opt/venv uv pip install "./ml[serve]" -r serving/pyproject.toml

FROM python:3.12-slim
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 10001 app
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app serving/app /app/app
USER app
EXPOSE 8001
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 CMD curl -fs http://localhost:8001/health || exit 1
CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8001", "--workers", "2"]
