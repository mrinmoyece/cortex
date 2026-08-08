# ─────────────────────────────────────────────────────────────────────────────
# Cortex Dockerfile — multi-stage build for minimal production image
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ── Builder stage ─────────────────────────────────────────────────────────────
FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --upgrade pip && \
    pip install --prefix=/install .

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM base AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN groupadd -r cortex && useradd -r -g cortex cortex
WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ src/
COPY config/ config/

USER cortex
EXPOSE 8000 8001

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
