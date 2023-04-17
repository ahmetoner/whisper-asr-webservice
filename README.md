![Release](https://img.shields.io/github/v/release/ahmetoner/whisper-asr-webservice.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/onerahmet/openai-whisper-asr-webservice.svg)
![Build](https://img.shields.io/github/actions/workflow/status/ahmetoner/whisper-asr-webservice/docker-publish.yml.svg)
![Licence](https://img.shields.io/github/license/ahmetoner/whisper-asr-webservice.svg)
# Whisper ASR Webservice

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification. For more details: [github.com/openai/whisper](https://github.com/openai/whisper/)

## Features
Current release (v1.1.0) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20230124](https://github.com/openai/whisper/releases/tag/v20230124)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)@[0.4.1](https://github.com/guillaumekln/faster-whisper/releases/tag/v0.4.1)
- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) (Coming soon)

## Usage

Whisper ASR Webservice now available on Docker Hub. You can find the latest version of this repository on docker hub for CPU and GPU.

Docker Hub: <https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice>

For CPU:

```sh
docker run -d -p 9000:9000 -e ASR_MODEL=base onerahmet/openai-whisper-asr-webservice:latest
```

For GPU:

```sh
docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base onerahmet/openai-whisper-asr-webservice:latest-gpu
```

For MacOS (CPU only):

GPU passthrough does not work on macOS due to fundamental design limitations of Docker. Docker actually runs containers within a LinuxVM on macOS. If you wish to run GPU-accelerated containers, I'm afraid Linux is your only option.

The `:latest` image tag provides both amd64 and arm64 architectures:

```sh
docker run -d -p 9000:9000 -e ASR_MODEL=base onerahmet/openai-whisper-asr-webservice:latest
```

```sh
# Interactive Swagger API documentation is available at http://localhost:9000/docs
```
![Swagger UI](https://github.com/ahmetoner/whisper-asr-webservice/blob/main/docs/assets/img/swagger-ui.png?raw=true)

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
pip3 install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch
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
docker-compose up --build
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

Returns a json with following fields:

- **text**: Contains the full transcript
- **segments**: Contains an entry per segment. Each entry  provides time stamps, transcript, token ids and other metadata
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
