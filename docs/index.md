Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition as well as speech translation and language identification.

## Features
Current release (v1.2.4) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20231117](https://github.com/openai/whisper/releases/tag/v20231117)
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)@[v0.10.0](https://github.com/SYSTRAN/faster-whisper/releases/tag/0.10.0)

## Quick Usage

=== ":octicons-file-code-16: `CPU`"

    ```sh
    docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
    ```

=== ":octicons-file-code-16: `GPU`"

    ```sh
    docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest-gpu
    ```

for more information:

- [Documentation/Run](https://ahmetoner.github.io/whisper-asr-webservice/run)
- [Docker Hub](https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice)
