# Whisper ASR Services

[![Codefresh build status]( https://g.codefresh.io/api/badges/pipeline/cbsiamlg/cbsiamlg%2Fwhisper-asr-webservice%2Fwhisper-asr-webservice?type=cf-1&key=eyJhbGciOiJIUzI1NiJ9.NWE2NjU5NDNjNjNkMzkwMDAxZjY4YmIy.TIFye2w47MUSn6ruP7AgrKo9PWqkwKlQvr1prmnFyJM)]( https://g.codefresh.io/pipelines/edit/new/builds?id=64b58efa9dca5a5131770ca0&pipeline=whisper-asr-webservice&projects=cbsiamlg%2Fwhisper-asr-webservice&projectId=64b58ee7b6d08d68e09da010)

## Overview

This project is a modification of an open source approach to containerizing the `faster-whisper` and `openai/whisper` ASR services for Splice. The modifications allow us to maintain security and make updates to model versions as well as store the images in our GAR. For more details: [github.com/openai/whisper](https://github.com/openai/whisper/).

## Features

AMLG updated release (v1.2.0) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20230124](https://github.com/openai/whisper/releases/tag/v20230124)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)@[0.6.0](https://github.com/guillaumekln/faster-whisper/releases/tag/v0.6.0)

## Usage

The service is available to run locally using the CPU Dockerfile. The GPU service is what is deployed.

For CPU (_for local use only_):

You will have to build the image locally:
`docker compose -f docker-compose.yml build` add `--dry-run` to test first.

The pull is not necessary as it's in your local docker.io
from the build above. Feel free to skip that step.

The endpoint will take a moment to become available so be patient.

```sh
docker pull whisper-asr-webservice-whisper-asr-webservice:cpu-latest
docker run -d -p 9000:9000 -e ASR_MODEL=medium.en -e ASR_ENGINE=faster_whisper whisper-asr-webservice-whisper-asr-webservice:cpu-latest
```

For GPU:

```sh
docker pull us-docker.pkg.dev/i-amlg-dev/sca/whisper-asr-webservice:latest
docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=medium.en -e ASR_ENGINE=faster_whisper us-docker.pkg.dev/i-amlg-dev/sca/whisper-asr-webservice:latest
```

For MacOS (CPU only):

GPU passthrough does not work on macOS due to fundamental design limitations of Docker. Docker actually runs containers within a LinuxVM on macOS. If you wish to run GPU-accelerated containers, I'm afraid Linux is your only option.

The `:latest` image tag provides both amd64 and arm64 architectures:

```sh
docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper whisper-asr-webservice-whisper-asr-webservice:cpu-latest
```

```sh
# Interactive Swagger API documentation is available at http://localhost:9000/docs
```

![Swagger UI](https://github.com/cbsiamlg/whisper-asr-webservice/blob/main/docs/assets/img/swagger-ui.png?raw=true)

Available ASR_MODELs are `tiny`, `base`, `small`, `medium`, `large`, `large-v1` and `large-v2`. Please note that `large` and `large-v2` are the same model.

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

## Run (Development Environment)

Install poetry with following command:

```sh
pip3 install poetry
```

Install torch with following command:

```sh
# just for GPU:
pip3 install torch==1.13.1+cu117 -f https://download.pytorch.org/whl/torch
```

Install packages:

```sh
poetry install
```

Starting the Webservice:

```sh
poetry run gunicorn --bind 0.0.0.0:9000 --workers 1 --timeout 0 app.webservice:app -k uvicorn.workers.UvicornWorker
```

With docker compose:

For CPU:

```sh
docker compose -f docker-compose.yml up
```

For GPU:

```sh
docker-compose up --build -f docker-compose.gpu.yml
```

## Quick start

After running the docker image interactive Swagger API documentation is available at [localhost:9000/docs](http://localhost:9000/docs)

There are 2 endpoints available:

- /asr (TXT, VTT, SRT, TSV, JSON)
- /detect-language

## Automatic Speech recognition service /asr

If you choose the **transcribe** task, transcribes the uploaded file. Both audio and video files are supported (as long as ffmpeg supports it).

Note that you can also upload video formats directly as long as they are supported by ffmpeg.

You can get TXT, VTT, SRT, TSV and JSON output as a file from /asr endpoint.

You can provide the language or it will be automatically recognized.

If you choose the **translate** task it will provide an English transcript no matter which language was spoken.

You can enable word level timestamps output by `word_timestamps` parameter (only with `Faster Whisper` for now).

Returns a json with following fields:

- **text**: Contains the full transcript
- **segments**: Contains an entry per segment. Each entry provides `timestamps`, `transcript`, `token ids`, `word level timestamps` and other metadata
- **language**: Detected or provided language (as a language code)

## Language detection service /detect-language

Detects the language spoken in the uploaded file. For longer files it only processes first 30 seconds.

Returns a json with following fields:

- **detected_language**
- **language_code**

## Build

Build .whl package

```sh
poetry build
```

Configuring the ASR Engine

```sh
export ASR_ENGINE=openai_whisper
```

or

```sh
export ASR_ENGINE=faster_whisper
```

Configuring the Model

```sh
export ASR_MODEL=base
```

## Docker Build

### For CPU

```sh
# Build Image
docker build -t whisper-asr-webservice .

# Run Container
docker run -d -p 9000:9000 whisper-asr-webservice
# or
docker run -d -p 9001:9000 -e ASR_MODEL=base whisper-asr-webservice3
```

### For GPU

```sh
# Build Image
docker build -f Dockerfile.gpu -t whisper-asr-webservice-gpu .

# Run Container
docker run -d --gpus all -p 9000:9000 whisper-asr-webservice-gpu
# or
docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice-gpu
```

## Cache

The ASR model is downloaded each time you start the container, using the large model this can take some time. If you want to decrease the time it takes to start your container by skipping the download, you can store the cache directory (/root/.cache/whisper) to an persistent storage. Next time you start your container the ASR Model will be taken from the cache instead of being downloaded again.

**Important this will prevent you from receiving any updates to the models.**

```sh
docker run -d -p 9000:9000 -e ASR_MODEL=large -v //c/tmp/whisper:/root/.cache/whisper onerahmet/openai-whisper-asr-webservice:latest
```

## Updates to the models

Going forward the poetry updates to whisper and faster-whisper will be run on a remote machine. The updated models will be stored in the GAR per the usual process. The docker images will be updated with the new models and pushed to the GAR. The images will be tagged with the model version. The latest tag will be updated to point to the latest model version. The cpu-latest is built and pushed as an AMD64 image and tagged as `cpu-latest` in the dev GAR only. You may not be able to build locally unless we add more logic for 