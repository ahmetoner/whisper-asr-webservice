FROM onerahmet/ffmpeg:n7.1 AS ffmpeg

FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui

FROM python:3.10-bookworm

LABEL org.opencontainers.image.source="https://github.com/ahmetoner/whisper-asr-webservice"

ENV POETRY_VENV=/app/.venv

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install --upgrade pip \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==2.1.3

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY --from=ffmpeg /usr/local/bin/ffmpeg /usr/local/bin/ffmpeg
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

# Copy poetry configuration files and install dependencies first to have them as cached docker layer
COPY pyproject.toml poetry.lock ./

RUN poetry config virtualenvs.in-project true

RUN --mount=type=cache,target=/root/.cache/pypoetry/artifacts \
    poetry install --extras cpu --no-root --no-interaction --no-ansi

# Copy the rest of the application code
COPY . .

# Install the project itself (dependencies should come installed from previous layer)
RUN poetry install --extras cpu --no-interaction --no-ansi

EXPOSE 9000

ENTRYPOINT ["whisper-asr-webservice"]
