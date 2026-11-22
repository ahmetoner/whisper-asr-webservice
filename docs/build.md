# Development

## Requirements

- Python 3.12 or 3.13
- [Poetry](https://python-poetry.org) 2.x
- [FFmpeg](https://ffmpeg.org) on the `PATH`
- For GPU support, an NVIDIA driver that supports CUDA 12.8 or newer

## Install

```shell
poetry install --extras cpu   # CPU build of PyTorch
poetry install --extras cuda  # PyTorch for CUDA 12.8
```

## Run

```shell
poetry run whisper-asr-webservice --host 0.0.0.0 --port 9000
```

Configure the service with the [environment variables](environmental-variables.md), for example
`ASR_ENGINE=faster_whisper ASR_MODEL=tiny poetry run whisper-asr-webservice`.

## Test and Lint

```shell
poetry run pytest
poetry run ruff check app tests
```

The tests replace the ASR engine with a stub, so they run without downloading models. Audio decoding tests are skipped
when FFmpeg is not installed.

## Build Docker Images

=== "CPU"

    ```shell
    docker build -t whisper-asr-webservice .
    docker run -d -p 9000:9000 whisper-asr-webservice
    ```

=== "GPU"

    ```shell
    docker build -f Dockerfile.gpu -t whisper-asr-webservice-gpu .
    docker run -d --gpus all -p 9000:9000 whisper-asr-webservice-gpu
    ```

=== "Docker Compose"

    ```shell
    docker compose up --build                            # CPU
    docker compose -f docker-compose.gpu.yml up --build  # GPU
    ```

    The Compose files mount `./app` into the container, so code changes take effect after a restart without a rebuild.

## Documentation

```shell
poetry run mkdocs serve
```

The site is served at <http://localhost:8000> and reloads on changes.

## Releases

Pushing a version tag publishes the Docker images, creates a GitHub release and deploys this documentation.
Pushing to the `debug` branch publishes the `debug` and `debug-gpu` images for testing.
