![Release](https://img.shields.io/github/v/release/ahmetoner/whisper-asr-webservice.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/onerahmet/openai-whisper-asr-webservice.svg)
![Build](https://img.shields.io/github/actions/workflow/status/ahmetoner/whisper-asr-webservice/docker-publish.yml.svg)
![Licence](https://img.shields.io/github/license/ahmetoner/whisper-asr-webservice.svg)

> 🎉 **Join our Discord Community!** Connect with other users, get help, and stay updated on the latest features: [https://discord.gg/4Q5YVrePzZ](https://discord.gg/4Q5YVrePzZ)

# Whisper ASR Box

Whisper ASR Box is a general-purpose speech recognition toolkit. Whisper Models are trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition as well as speech translation and language identification.

## Features

Current release (v1.9.1) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20250625](https://github.com/openai/whisper/releases/tag/v20250625)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.1](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.1.1)
- [whisperX](https://github.com/m-bain/whisperX)@[v3.4.2](https://github.com/m-bain/whisperX/releases/tag/v3.4.2)

## Quick Usage

### CPU

```shell
docker run -d -p 9000:9000 \
  -e ASR_MODEL=base \
  -e ASR_ENGINE=openai_whisper \
  onerahmet/openai-whisper-asr-webservice:latest
```

### GPU

```shell
docker run -d --gpus all -p 9000:9000 \
  -e ASR_MODEL=base \
  -e ASR_ENGINE=openai_whisper \
  onerahmet/openai-whisper-asr-webservice:latest-gpu
```

#### Cache

To reduce container startup time by avoiding repeated downloads, you can persist the cache directory:

```shell
docker run -d -p 9000:9000 \
  -v $PWD/cache:/root/.cache/ \
  onerahmet/openai-whisper-asr-webservice:latest
```

## Key Features

- Multiple ASR engines support (OpenAI Whisper, Faster Whisper, WhisperX)
- Multiple output formats (text, JSON, VTT, SRT, TSV)
- Word-level timestamps support
- Voice activity detection (VAD) filtering
- Speaker diarization (with WhisperX)
- FFmpeg integration for broad audio/video format support
- GPU acceleration support
- Configurable model loading/unloading
- REST API with Swagger documentation

## Environment Variables

Key configuration options:

- `ASR_ENGINE`: Engine selection (openai_whisper, faster_whisper, whisperx)
- `ASR_MODEL`: Model selection (tiny, base, small, medium, large-v3, etc.)
- `ASR_MODEL_PATH`: Custom path to store/load models
- `ASR_DEVICE`: Device selection (cuda, cpu)
- `MODEL_IDLE_TIMEOUT`: Timeout for model unloading

## Documentation

For complete documentation, visit:
[https://ahmetoner.github.io/whisper-asr-webservice](https://ahmetoner.github.io/whisper-asr-webservice)

## Development

```shell
# Install poetry v2.X
pip3 install poetry

# Install dependencies for cpu
poetry install --extras cpu

# Install dependencies for cuda
poetry install --extras cuda

# Run service
poetry run whisper-asr-webservice --host 0.0.0.0 --port 9000
```

After starting the service, visit `http://localhost:9000` or `http://0.0.0.0:9000` in your browser to access the Swagger UI documentation and try out the API endpoints.

## Credits

- This software uses libraries from the [FFmpeg](http://ffmpeg.org) project under the [LGPLv2.1](http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
