# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Builder: resolve dependencies into a self-contained venv.
# Kept separate so pip, its caches and any build tooling never reach the
# runtime image.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build

# Only what the install needs. `main.py` is included because pyproject declares
# it via py-modules for the `llm-extraction` console script.
COPY pyproject.toml main.py ./
COPY src ./src

# The API server plus the OpenRouter provider — what the compose services use.
# Without a key at runtime the app still falls back to the offline mock adapter.
RUN pip install ".[api,openrouter]"


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Unprivileged. The API writes nothing; the CLI writes only to a bind mount,
# and compose overrides the uid there to match the host.
RUN useradd --create-home --uid 10001 app

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# `src` is importable from site-packages, so only the CLI entry point and the
# default input are copied. quotes.json is a convenience default: compose
# bind-mounts over it so the evaluator can swap in their own.
COPY --chown=app:app main.py quotes.json ./

RUN mkdir -p /app/outputs && chown app:app /app/outputs

USER app
EXPOSE 8000

# urllib rather than curl: python:3.12-slim ships no curl, and adding one just
# for a healthcheck is a package and a CVE surface we do not need.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"]

# 0.0.0.0, not 127.0.0.1 — bound to loopback the published port is unreachable
# from the host.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
