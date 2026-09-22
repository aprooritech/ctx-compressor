# syntax=docker/dockerfile:1

# ---- build stage -----------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app
RUN python -m venv /opt/venv

COPY pyproject.toml README.md main.py ./
COPY src ./src
RUN pip install --no-cache-dir .

# Bake the BPE vocabulary into the image: the container then never needs to
# reach the tiktoken CDN at startup.
RUN mkdir -p "$TIKTOKEN_CACHE_DIR" \
    && python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# ---- runtime stage ---------------------------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    DB_PATH=/app/data/proxy.db

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
COPY main.py ./
COPY src ./src

RUN useradd --create-home --uid 10001 proxy \
    && mkdir -p /app/data \
    && chown -R proxy:proxy /app /opt/tiktoken

USER proxy
EXPOSE 8000

# Cheap liveness probe - /healthz additionally probes the backends.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
