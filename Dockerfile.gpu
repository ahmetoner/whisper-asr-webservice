FROM swaggerapi/swagger-ui:v5.9.1 AS swagger-ui
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV PYTHON_VERSION=3.10
ENV POETRY_VENV=/app/.venv

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python3 && \
    ln -s -f /usr/bin/python${PYTHON_VERSION} /usr/bin/python && \
    ln -s -f /usr/bin/pip3 /usr/bin/pip

RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==1.6.1

ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /app

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.in-project true
RUN poetry install --no-root

COPY . .
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui.css swagger-ui-assets/swagger-ui.css
COPY --from=swagger-ui /usr/share/nginx/html/swagger-ui-bundle.js swagger-ui-assets/swagger-ui-bundle.js

RUN poetry install
RUN $POETRY_VENV/bin/pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch

EXPOSE 9000

CMD gunicorn --bind 0.0.0.0:9000 --workers 1 --timeout 0 app.webservice:app -k uvicorn.workers.UvicornWorker
