FROM onerahmet/ffmpeg:n9.1 AS ffmpeg

FROM swaggerapi/swagger-ui:v5.35.1 AS swagger-ui

FROM python:3.12-slim-bookworm AS base

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
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip

FROM base AS build

RUN python3 -m venv /opt/poetry \
    && /opt/poetry/bin/pip install --no-cache-dir poetry==2.5.2

WORKDIR /app

COPY pyproject.toml poetry.lock poetry.toml ./
RUN /opt/poetry/bin/poetry config virtualenvs.in-project true \
    && /opt/poetry/bin/poetry config virtualenvs.options.no-pip true \
    && /opt/poetry/bin/poetry install --extras cpu --without dev --no-root

COPY README.md ./
COPY app ./app
RUN /opt/poetry/bin/poetry install --only-root

FROM base

ENV PATH="${PATH}:/app/.venv/bin"

WORKDIR /app

COPY --from=build /app/.venv /app/.venv
COPY . .
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=ffmpeg /usr/local/share/doc/ffmpeg /usr/local/share/doc/ffmpeg
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5m --retries=3 \
    CMD ["python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:9000/health', timeout=4)"]

ENTRYPOINT ["whisper-asr-webservice"]
