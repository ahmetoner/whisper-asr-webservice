Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition as well as speech translation and language identification.

## Features

Current release (v1.8.2) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/releases/tag/v20240930)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v1.1.0](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.1.0)
- [whisperX](https://github.com/m-bain/whisperX)@[v3.1.1](https://github.com/m-bain/whisperX/releases/tag/v3.1.1)

## Quick Usage

=== ":octicons-file-code-16: `CPU`"

    ```shell
    docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `GPU`"

    ```shell
    docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest-gpu
    ```

for more information:

- [Documentation/Run](https://ahmetoner.github.io/whisper-asr-webservice/run)
- [Docker Hub](https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice)

## Credits

- This software uses libraries from the [FFmpeg](http://ffmpeg.org) project under the [LGPLv2.1](http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html)
