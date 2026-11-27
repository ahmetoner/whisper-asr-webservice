FROM swaggerapi/swagger-ui:v5.35.1 AS swagger-ui

FROM python:3.12-slim-trixie AS base

LABEL org.opencontainers.image.title="Whisper ASR Webservice" \
      org.opencontainers.image.description="Self-hosted speech recognition API based on OpenAI Whisper" \
      org.opencontainers.image.url="https://github.com/ahmetoner/whisper-asr-webservice" \
      org.opencontainers.image.documentation="https://ahmetoner.github.io/whisper-asr-webservice" \
      org.opencontainers.image.source="https://github.com/ahmetoner/whisper-asr-webservice" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.authors="Ahmet Öner"

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq upgrade -y \
    && apt-get -qq install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip uninstall -y pip

FROM base AS build

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md LICENCE ./
COPY app ./app
RUN uv sync --locked --no-dev

FROM base

ENV PATH="${PATH}:/app/.venv/bin" \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=4)"]

ENTRYPOINT ["whisper-asr-webservice"]
