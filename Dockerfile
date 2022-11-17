FROM python:3.9.9-slim

ENV POETRY_VERSION=1.2.0
ENV POETRY_VENV=/app/.venv

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

ARG TARGETPLATFORM
RUN if [ "$TARGETPLATFORM" = "linux/amd64" ]; then $POETRY_VENV/bin/pip install torch==1.13.0 -f https://download.pytorch.org/whl/cpu; fi;
RUN if [ "$TARGETPLATFORM" = "linux/arm64" ]; then $POETRY_VENV/bin/pip install torch==1.13.0; fi;

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY . /app

RUN poetry config virtualenvs.in-project true
RUN poetry install

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9000", "--workers", "1", "--timeout", "0", "whisper_asr.webservice:app", "-k", "uvicorn.workers.UvicornWorker"]