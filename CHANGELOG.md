Changelog
=========

Unreleased
----------

[1.9.1] (2025-07-01)
--------------------

### Fixed

- Fixed Whisperx diarization pipeline initialization
- Fixed Whisperx language detection

[1.9.0] (2025-06-29)
--------------------

### Changed

- Upgraded
  - Poetry to v2.1.3
  - [openai/whisper](https://github.com/openai/whisper)@[v20250625](https://github.com/openai/whisper/releases/tag/v20250625)
  - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) to [v1.1.1](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.1.1)
  - [whisperX](https://github.com/m-bain/whisperX)@[v3.4.2](https://github.com/m-bain/whisperX/releases/tag/v3.4.2)
  - torch to v2.7.1
  - torchaudio to v2.7.1
  - numpy to v2.2.6
  - fastapi to v0.115.14
  - uvicorn to v0.35.0
  - numba to v0.61.2

[1.8.2] (2025-02-18)
--------------------

### Changed

- Reduced GPU image size by using `nvidia/cuda:12.6.3-base-ubuntu22.04`

[1.8.1] (2025-02-18)
--------------------

### Fixed

- Fixed issues with Torch CUDA and cuDNN
- Updated Torch and Torchaudio dependencies for multi-architecture support

[1.8.0] (2025-02-17)
--------------------

### Added

- Added support for [whisperX](https://github.com/m-bain/whisperX)@[v3.1.1](https://github.com/m-bain/whisperX/releases/tag/v3.1.1)

### Changed

- Upgraded Cuda GPU image to v12.6.3
- Upgraded dependencies
  - torch to v2.6.0
  - fastapi to v0.115.8
  - llvmlite to v0.44.0
  - numba to v0.61.0
  - ruff to v0.9.6
  - black to v25.1.0
  - mkdocs-material to v9.6.4
  - pymdown-extensions to v10.14.3

[1.7.1] (2024-12-18)
--------------------

### Fixed

- Fix JSON serialization of segments due to Faster Whisper v1.1.0 changes

[1.7.0] (2024-12-17)
--------------------

### Added

  - Timeout configured to allow model to be unloaded when idle
  - Added detection confidence to langauge detection endpoint
  - Set mel generation to adjust n_dims automatically to match the loaded model
  - Refactor classes, Add comments, implement abstract methods, and add factory method for engine selection

### Changed

- Upgraded
  - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) to [v1.1.0](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.1.0)
  - uvicorn to v0.34.0
  - tqdm to v4.67.1
  - python-multipart to v0.0.20
  - fastapi to v0.115.6
  - pytest to v8.3.4
  - ruff to v0.8.3
  - black to v24.10.0
  - mkdocs to v1.6.1
  - mkdocs-material to v9.5.49
  - pymdown-extensions to v10.12

[1.6.0] (2024-10-06)
--------------------

### Changed

- Upgraded
  - [openai/whisper](https://github.com/openai/whisper)@[v20240930](https://github.com/openai/whisper/releases/tag/v20240930)
  - fastapi to v0.115.0
  - uvicorn to v0.31.0
  - tqdm to v4.66.5
  - python-multipart to v0.0.12

[1.5.0] (2024-07-04)
--------------------

### Changed

- Upgraded
  - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) to [v1.0.3](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.0.3)
  - fastapi to v0.111.0
  - uvicorn to v0.30.1
  - gunicorn to v22.0.0
  - tqdm to v4.66.4
  - llvmlite to v0.43.0
  - numba to v0.60.0

[1.4.1] (2024-04-17)
--------------------

### Changed

- Upgraded torch to v1.13.1

[1.4.0] (2024-04-17)
--------------------

### Changed

- Upgraded
  - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) to [v1.0.1](https://github.com/SYSTRAN/faster-whisper/releases/tag/v1.0.1)
  - fastapi to v0.110.1
  - uvicorn to v0.29.0
  - gunicorn to v21.2.0
  - tqdm to v4.66.2
  - python-multipart to v0.0.9
  - llvmlite to v0.42.0
  - numba to v0.59.1

[1.3.0] (2024-02-15)
--------------------

### Added

- Compiled and added FFmpeg without LGPL libraries for license compliance

[1.2.4] (2023-11-27)
--------------------

### Changed

- Upgraded
  - [openai/whisper](https://github.com/openai/whisper) to [v20231117](https://github.com/openai/whisper/releases/tag/v20231117)
  - [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) to [v0.10.0](https://github.com/SYSTRAN/faster-whisper/releases/tag/v0.10.0)

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
- mp3 support by using FFmpeg instead of librosa in #8
- add language detection endpoint in #9

[1.9.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.9.1
[1.9.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.9.0
[1.8.2]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.8.2
[1.8.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.8.1
[1.8.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.8.0
[1.7.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.7.1
[1.7.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.7.0
[1.6.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.6.0
[1.5.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.5.0
[1.4.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.4.1
[1.4.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.4.0
[1.3.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.3.0
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
