FROM onerahmet/ffmpeg:n9.1 AS ffmpeg

FROM swaggerapi/swagger-ui:v5.35.1 AS swagger-ui

FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.source="https://github.com/ahmetoner/whisper-asr-webservice"

# Apply Debian security updates and upgrade the base image's pip, whose bundled urllib3 and certifi are outdated
RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -U pip

ENV POETRY_VENV=/app/.venv

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==2.5.2

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY . .
COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

RUN poetry config virtualenvs.in-project true
RUN poetry install --extras cpu

EXPOSE 9000

ENTRYPOINT ["whisper-asr-webservice"]
