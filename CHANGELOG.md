Changelog
=========

Unreleased
----------

### Added

- API test suite and a CI workflow running ruff and pytest
- `ASR_BEAM_SIZE`, `ASR_BATCH_SIZE` and `ASR_VAD_FILTER` environment variables
- `/health` endpoint for container health checks, public even when `API_KEY` is set
- Optional API key authentication with the `API_KEY` environment variable
- GitHub release is created automatically when a version tag's Docker images are published successfully
- Docker images can run as a non-root user (`--user`) with `XDG_CACHE_HOME` pointing to a writable cache directory; `ASR_MODEL_PATH` now defaults to `$XDG_CACHE_HOME/whisper`
- Trivy vulnerability scans on pull requests and before publishing images: fixable HIGH/CRITICAL findings fail the pipeline, MEDIUM findings produce a warning
- Per-architecture Docker image tags (`<version>-amd64`, `<version>-arm64`, `<version>-gpu-amd64`); arm64 images are built on native runners instead of under emulation

### Changed

- Upgraded
  - Python requirement to >=3.12,<3.14
  - Poetry to v2.5.2
  - PyTorch to v2.8.0 (CUDA builds now use CUDA 12.8, adding RTX 50xx / Blackwell `sm_120` support)
  - whisperx to v3.8.6 and pyannote-audio to v4.1.1
  - fastapi to v0.165.0
  - uvicorn to v0.54.0
  - python-multipart to v0.0.34
  - tqdm to v4.70.1
  - llvmlite to v0.50.0
  - numba to v0.68.1
  - GitHub Actions to their latest major versions
- WhisperX diarization now uses `pyannote/speaker-diarization-community-1`; accept its conditions on Hugging Face
- Only the selected ASR engine is imported at startup, reducing startup time and memory use
- Docker images use the FFmpeg n9.1 build layer
- Refreshed locked dependencies to pick up security fixes (aiohttp, certifi, filelock, lightning, nltk, pillow, urllib3 and more)
- CPU Docker image is based on `python:3.12-slim-bookworm`
- Docker images include the FFmpeg license (LGPL v2.1+) at `/usr/local/share/doc/ffmpeg`
- Docker images no longer include development dependencies, and dependencies are installed in their own layer so rebuilds after code changes are faster
- Rewrote the README and documentation, adding the API reference, a complete configuration table and the development guide

### Fixed

- WhisperX crash loop at startup (`Weights only load failed`) caused by lightning 2.6 with pyannote-audio 3.x
- Model failing to reload after `MODEL_IDLE_TIMEOUT` unloaded it, especially with WhisperX
- `initial_prompt` causing HTTP 500 with the WhisperX engine
- WhisperX diarization ignoring `min_speakers` and `max_speakers`
- `/detect-language` returning the confidence of the wrong language with the openai_whisper engine
- Empty transcripts for M4A/MP4/MOV files that store their index at the end of the file
- Undecodable, empty or malformed raw (`encode=false`) uploads now return HTTP 400
- `/asr` responses now use a `Content-Type` matching the requested `output` format
- Swagger UI showing a text field instead of a file picker for `audio_file` in Docker images
- Server becoming unresponsive, including `/docs`, while a transcription is running
- Missing contact URL in the OpenAPI metadata

### Security

- CPU Docker image base switched to `python:3.12-slim-bookworm`, reducing OS package findings reported by Trivy from 21 critical / 688 high to 2 critical / 99 high
- CPU Docker image applies Debian security updates at build time
- Upgraded the system pip in the CPU Docker image, which bundles urllib3 and certifi, fixing
  - GHSA-6q48-6c7w-9gw2
  - CVE-2025-50181
  - CVE-2025-66418
  - CVE-2025-66471
  - CVE-2026-21441
  - CVE-2026-44431
  - CVE-2026-97687
  - CVE-2026-97689
- Upgraded the FFmpeg build layer from n7.1 to n9.1, which includes upstream FFmpeg security fixes
- Upgraded torch to v2.8.0, fixing
  - CVE-2025-3730
- Upgraded aiohttp to v3.14.5, fixing
  - CVE-2026-111063
  - CVE-2026-111064
  - CVE-2026-111065
  - CVE-2026-111066
  - CVE-2026-111748
  - CVE-2026-111749
  - CVE-2026-111750
  - CVE-2026-123817
  - CVE-2026-128124
  - CVE-2026-128125
  - CVE-2026-128126
  - CVE-2026-128127
  - CVE-2026-128128
  - CVE-2026-128129
  - CVE-2026-128130
  - CVE-2026-128131
  - CVE-2026-128132
- Upgraded certifi to v2026.11.20, fixing
  - GHSA-6q48-6c7w-9gw2
- Upgraded filelock to v4.3.0, fixing
  - CVE-2026-112907
  - CVE-2026-112908
  - CVE-2026-112909
  - CVE-2026-112910
  - CVE-2026-112911
  - CVE-2026-112913
  - CVE-2026-112915
  - CVE-2026-112935
  - CVE-2026-112936
  - CVE-2026-112938
  - CVE-2026-112939
  - CVE-2026-112940
- Upgraded lightning to v2.6.6, fixing
  - CVE-2026-58659
- Upgraded mako to v1.4.3, fixing
  - CVE-2026-102991
- Upgraded multidict to v7.2.1, fixing
  - CVE-2026-104874
- Upgraded nltk to v3.10.4, fixing
  - CVE-2026-12876
  - CVE-2026-71513
  - CVE-2026-71514
  - CVE-2026-78680
  - CVE-2026-78681
  - CVE-2026-78682
  - CVE-2026-79657
  - CVE-2026-79674
  - CVE-2026-79675
  - CVE-2026-79676
  - CVE-2026-80206
  - CVE-2026-81722
  - CVE-2026-81723
  - CVE-2026-81724
  - CVE-2026-81725
  - CVE-2026-81726
  - CVE-2026-81727
  - CVE-2026-126295
  - CVE-2026-126296
  - CVE-2026-126297
  - CVE-2026-126298
  - CVE-2026-126299
  - CVE-2026-126300
  - CVE-2026-126301
  - CVE-2026-126302
  - CVE-2026-126303
  - CVE-2026-126304
  - CVE-2026-128134
  - CVE-2026-128135
  - CVE-2026-128136
  - CVE-2026-128137
  - CVE-2026-128139
  - CVE-2026-128140
  - CVE-2026-129221
  - CVE-2026-129222
  - CVE-2026-129223
  - CVE-2026-129224
  - CVE-2026-129225
  - CVE-2026-129226
  - CVE-2026-129227
  - CVE-2026-129228
  - CVE-2026-129229
  - CVE-2026-129230
  - CVE-2026-129231
  - CVE-2026-129232
  - CVE-2026-129233
  - CVE-2026-129234
  - GHSA-j456-xh4h-cpf2
- Upgraded pillow to v13.0.0, fixing
  - CVE-2026-96636
  - CVE-2026-96637
  - CVE-2026-96638
  - CVE-2026-96639
  - CVE-2026-96640
  - CVE-2026-96641
  - CVE-2026-96642
  - CVE-2026-96643
  - CVE-2026-96644
  - CVE-2026-96778
  - CVE-2026-112390
  - CVE-2026-112391
  - CVE-2026-112392
  - CVE-2026-112393
  - CVE-2026-112394
  - CVE-2026-112395
  - CVE-2026-112396
  - CVE-2026-112397
  - CVE-2026-112398
  - CVE-2026-112399
  - CVE-2026-112400
  - CVE-2026-112401
  - CVE-2026-112402
  - CVE-2026-112403
  - CVE-2026-112404
  - CVE-2026-112405
  - CVE-2026-112406
  - CVE-2026-112891
  - CVE-2026-112892
  - CVE-2026-112893
  - CVE-2026-112894
  - CVE-2026-112895
  - CVE-2026-112896
  - CVE-2026-112897
  - CVE-2026-112898
  - CVE-2026-112899
  - CVE-2026-112900
  - CVE-2026-112901
  - CVE-2026-112902
  - CVE-2026-114501
  - CVE-2026-128066
  - CVE-2026-128067
  - GHSA-78g4-xqwv-q3g8
- Upgraded setuptools to v84.2.1, fixing
  - GHSA-xmj3-9gf7-h52w
- Upgraded urllib3 to v2.8.0, fixing
  - CVE-2026-97687
  - CVE-2026-97688
  - CVE-2026-97689
- Upgraded openssl to v3.0.13-0ubuntu3.16 in the GPU Docker image, fixing
  - CVE-2026-45447
  - CVE-2026-84782
- Upgraded apt to v2.8.4ubuntu0.1 in the GPU Docker image, fixing
  - CVE-2026-109518

[1.10.0] (2026-08-09)
---------------------

### Changed

- Upgrade CUDA image to 12.9.2-cudnn-runtime-ubuntu24.04
- Upgraded
  - Python to v3.12
  - Poetry to v2.4.1
  - fastapi to v0.141.1
  - uvicorn to v0.52.1
  - python-multipart to v0.0.32
  - faster-whisper to v1.2.1
  - tqdm to v4.70.0
  - llvmlite to v0.48.0
  - numba to v0.66.0
  - whisperx to v3.4.5

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

[1.10.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.10.0
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
