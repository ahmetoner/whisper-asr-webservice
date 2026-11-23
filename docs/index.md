# Whisper ASR Webservice

Whisper ASR Webservice is a self-hosted REST API for speech recognition, translation and language identification
based on [OpenAI Whisper](https://github.com/openai/whisper). It runs as a Docker container on CPU or NVIDIA GPU.

## Features

- OpenAI-compatible API: works with the OpenAI SDKs and any application that lets you set the OpenAI base URL
- Three engines: OpenAI Whisper, Faster Whisper and WhisperX
- Transcription in every language Whisper supports, and translation into English
- Language detection with a confidence score
- Output as plain text, JSON, WebVTT, SRT or TSV
- Word-level timestamps, voice activity detection (VAD) and speaker diarization
- Accepts any audio or video file FFmpeg can decode, decoded in-process with PyAV
- Optional API key authentication
- Unloads the model when idle to free memory
- Interactive API documentation (Swagger UI)

## Quick Start

=== "CPU"

    ```shell
    docker run -d -p 9000:9000 \
      -e ASR_ENGINE=faster_whisper \
      -e ASR_MODEL=base \
      onerahmet/openai-whisper-asr-webservice:latest
    ```

=== "GPU"

    ```shell
    docker run -d --gpus all -p 9000:9000 \
      -e ASR_ENGINE=faster_whisper \
      -e ASR_MODEL=base \
      onerahmet/openai-whisper-asr-webservice:latest-gpu
    ```

Open <http://localhost:9000/docs> to use the API from your browser, or send a request directly:

```shell
curl http://localhost:9000/v1/audio/transcriptions -F model=whisper-1 -F file=@audio.mp3
```

## Engines

| Engine                                                       | `ASR_ENGINE`               | Notes                                                    |
|--------------------------------------------------------------|----------------------------|----------------------------------------------------------|
| [OpenAI Whisper](https://github.com/openai/whisper)          | `openai_whisper` (default) | Reference implementation                                 |
| [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)  | `faster_whisper`           | Faster and lighter on memory; VAD and batched inference  |
| [WhisperX](https://github.com/m-bain/whisperX)               | `whisperx`                 | Word alignment and speaker diarization                   |

## Next Steps

- [Installation](run.md): Docker images, GPU setup and model caching
- [OpenAI API](openai.md): the OpenAI-compatible endpoints and SDK examples
- [Legacy API](endpoints.md): the original endpoints, parameters and response formats
- [Configuration](environmental-variables.md): all environment variables
- [Development](build.md): running from source, tests and building images

## Support

Report bugs and request features on [GitHub Issues](https://github.com/ahmetoner/whisper-asr-webservice/issues).
For questions and discussion, join the [Discord server](https://discord.gg/4Q5YVrePzZ).

## License

Released under the [MIT License](licence.md). Audio is decoded with [PyAV](https://pyav.org), which bundles
[FFmpeg](https://ffmpeg.org) libraries licensed under the [LGPL v3](https://www.gnu.org/licenses/lgpl-3.0.html).
