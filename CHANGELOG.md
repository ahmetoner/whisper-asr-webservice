Changelog
=========

[1.2.4] (2023-11-27)
--------------------

### Changed

- Upgraded
    - [openai/whisper](https://github.com/openai/whisper) to [v20231117](https://github.com/openai/whisper/releases/tag/v20231117)
    - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) to [v0.10.0](https://github.com/SYSTRAN/faster-whisper/releases/tag/0.10.0)

[1.2.3] (2023-11-07)
--------------------

### Changed

- Upgraded
    - [openai/whisper](https://github.com/openai/whisper) to [v20231106](https://github.com/openai/whisper/releases/tag/v20231106)

[1.2.2] (2023-11-03)
--------------------

### Fixed

- Fixed `swagger-ui` rendering issues by upgrading to `v5.9.1`, fixes #153 and #154

[1.2.1] (2023-11-03)
--------------------

### Enabled

- Enabled `vad_filter` for `faster-whisper` engine

### Changed

- Changed misspelling in "Word level timestamps"
- Removed unused unidecode dependency
- Upgraded
    - uvicorn to v0.23.2
    - gunicorn to v21.0.1
    - tqdm to v4.66.1
    - python-multipart to v0.0.6
    - fastapi to v0.104.1
    - llvmlite to v0.41.1
    - numba to v0.58.0

[1.2.0] (2023-10-01)
--------------------

### Changed

- Upgraded
    - [openai/whisper](https://github.com/openai/whisper) to [v20230918](https://github.com/openai/whisper/releases/tag/v20230918)
    - [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper) to [v0.9.0](https://github.com/guillaumekln/faster-whisper/releases/tag/v0.9.0)

### Updated

- Updated model conversion method (for Faster Whisper) to use Hugging Face downloader
- Updated default model paths to `~/.cache/whisper` or `/root/.cache/whisper`.
    - For customization, modify the `ASR_MODEL_PATH` environment variable.
    - Ensure Docker volume is set for the corresponding directory to use caching.
      ```bash
      docker run -d -p 9000:9000 -e ASR_MODEL_PATH=/data/whisper -v $PWD/yourlocaldir:/data/whisper onerahmet/openai-whisper-asr-webservice:latest
      ```
- Removed the `triton` dependency from `poetry.lock` to ensure the stability of the pipeline for `ARM-based` Docker images

[1.1.1] (2023-05-29)
--------------------

### Changed

- 94 gpus that don't support float16 in #103
- Update compute type in #108
- Add word level functionality for Faster Whisper in #109

[1.1.0] (2023-04-17)
--------------------

### Changed

- Docs in #72
- Fix language code typo in #77
- Adds support for FasterWhisper in #81
- Add an optional param to skip the encoding step in #82
- Faster whisper in #92

[1.0.6] (2023-02-05)
--------------------

### Changed

- Update README.md in #58
- 68 update the versions in #69
- Fix gunicorn run command and remove deprecated poetry run script in #70
- Move torch installation method into the pyproject.toml file in #71
- Add prompt to ASR in #66

[1.0.5] (2022-12-08)
--------------------

### Changed

- 43 make swagger doc not depend on internet connection in #52
- Add new large model v2 in #53

[1.0.4] (2022-11-28)
--------------------

### Changed

- 43 make swagger doc not depend on internet connection in #51
- Anally retentively fixed markdown linting warnings in README. Sorry. in #48
- Explicit macOS readme with explanation for no-GPU [closes #44] in #47

[1.0.3-beta] (2022-11-17)
-------------------------

### Changed

- Combine transcribe endpoints in #36
- Add multi worker support with gunicorn in #37
- Add multi platform (amd & arm) support in #39
- Upgrade Cuda version to 11.7 in #40
- Lock to the latest whisper version (eff383) in #41

[1.0.2-beta] (2022-10-04)
-------------------------

### Changed

- add mutex lock to the model in #19
- Subtitles in #21
- Add gpu support and create Docker image for cuda with GitHub flow in #22

[1.0.1-beta] (2022-09-27)
-------------------------

### Changed

- Init GitHub runners in #10
- Lock Whisper dependency with b4308... revision number to prevent build crashes in #15

[1.0.0-beta] (2022-09-25)
-------------------------

### Changed

- Docker init in #1
- Create LICENCE in #2
- Fastapi init in #3
- Avoid temp file in #4
- Translate init in #5
- mp3 support by using ffmpeg instead of librosa in #8
- add language detection endpoint in #9

[1.2.4]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.2.4
[1.2.3]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.2.3
[1.2.2]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.2.2
[1.2.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.2.1
[1.2.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.2.0
[1.1.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.1.1
[1.1.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.1.0
[1.0.6]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.6
[1.0.5]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.5
[1.0.4]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.4
[1.0.3-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.3-beta
[1.0.2-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.2-beta
[1.0.1-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.1-beta
[1.0.0-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/1.0.0-beta
