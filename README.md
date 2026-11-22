# Whisper ASR Webservice

[![Release](https://img.shields.io/github/v/release/ahmetoner/whisper-asr-webservice.svg)](https://github.com/ahmetoner/whisper-asr-webservice/releases)
[![Docker Pulls](https://img.shields.io/docker/pulls/onerahmet/openai-whisper-asr-webservice.svg)](https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice)
[![Test](https://img.shields.io/github/actions/workflow/status/ahmetoner/whisper-asr-webservice/test.yml?label=test)](https://github.com/ahmetoner/whisper-asr-webservice/actions/workflows/test.yml)
[![License](https://img.shields.io/github/license/ahmetoner/whisper-asr-webservice.svg)](LICENCE)

A self-hosted REST API for speech recognition, translation and language identification based on
[OpenAI Whisper](https://github.com/openai/whisper). It runs as a Docker container on CPU or NVIDIA GPU.

## Features

- Three engines: OpenAI Whisper, Faster Whisper and WhisperX
- Transcription in every language Whisper supports, and translation into English
- Language detection with a confidence score
- Output as plain text, JSON, WebVTT, SRT or TSV
- Word-level timestamps, voice activity detection (VAD) and speaker diarization
- Accepts any audio or video file FFmpeg can decode
- Optional API key authentication
- Unloads the model when idle to free memory
- Interactive API documentation (Swagger UI)

## Quick Start

CPU (amd64 and arm64):

```shell
docker run -d -p 9000:9000 \
  -e ASR_ENGINE=faster_whisper \
  -e ASR_MODEL=base \
  onerahmet/openai-whisper-asr-webservice:latest
```

NVIDIA GPU (amd64, requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)):

```shell
docker run -d --gpus all -p 9000:9000 \
  -e ASR_ENGINE=faster_whisper \
  -e ASR_MODEL=base \
  onerahmet/openai-whisper-asr-webservice:latest-gpu
```

Open <http://localhost:9000/docs> to use the API from your browser, or send a request directly:

```shell
curl -F "audio_file=@audio.mp3" "http://localhost:9000/asr?output=json"
```

## Engines

| Engine                                                       | `ASR_ENGINE`               | Notes                                                    |
|--------------------------------------------------------------|----------------------------|----------------------------------------------------------|
| [OpenAI Whisper](https://github.com/openai/whisper)          | `openai_whisper` (default) | Reference implementation                                 |
| [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)  | `faster_whisper`           | Faster and lighter on memory; VAD and batched inference  |
| [WhisperX](https://github.com/m-bain/whisperX)               | `whisperx`                 | Word alignment and speaker diarization                   |

## Configuration

The most common settings are listed below. See the [configuration reference](https://ahmetoner.github.io/whisper-asr-webservice/environmental-variables/) for all of them.

| Variable             | Default            | Description                                                 |
|----------------------|--------------------|-------------------------------------------------------------|
| `ASR_ENGINE`         | `openai_whisper`   | `openai_whisper`, `faster_whisper` or `whisperx`            |
| `ASR_MODEL`          | `base`             | Model name, e.g. `tiny`, `small`, `large-v3`, `turbo`       |
| `ASR_DEVICE`         | `cuda` if present  | `cuda` or `cpu`                                             |
| `MODEL_IDLE_TIMEOUT` | `0`                | Seconds of inactivity before the model is unloaded          |
| `API_KEY`            | unset              | Require this key in the `X-API-Key` header                  |
| `HF_TOKEN`           | unset              | Hugging Face token, required for WhisperX diarization       |

To avoid downloading the model on every start, mount a volume at `/root/.cache`:

```shell
docker run -d -p 9000:9000 -v $PWD/cache:/root/.cache onerahmet/openai-whisper-asr-webservice:latest
```

## Documentation

Full documentation is available at <https://ahmetoner.github.io/whisper-asr-webservice>.

## Development

Requires Python 3.12 or 3.13 and [Poetry](https://python-poetry.org) 2.x.

```shell
poetry install --extras cpu       # or --extras cuda
poetry run whisper-asr-webservice --host 0.0.0.0 --port 9000
poetry run pytest
```

See the [development guide](https://ahmetoner.github.io/whisper-asr-webservice/build/) for building Docker images and the documentation.

## Support

Report bugs and request features on [GitHub Issues](https://github.com/ahmetoner/whisper-asr-webservice/issues).
For questions and discussion, join the [Discord server](https://discord.gg/4Q5YVrePzZ).

## License

Released under the [MIT License](LICENCE). The Docker images include [FFmpeg](https://ffmpeg.org), licensed under the
[LGPL v2.1](https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html).
