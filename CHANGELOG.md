Changelog
=========

Unreleased
----------

### Updated

- Updated model conversion method (for Faster Whisper) to use Hugging Face downloader

### Changed

- Upgraded
  - OpenAI Whisper to v20230918

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

[1.1.1]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.1.1

[1.1.0]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.1.0

[1.0.6]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.6

[1.0.5]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.5

[1.0.4]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.4

[1.0.3-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.3-beta

[1.0.2-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.2-beta

[1.0.1-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/v1.0.1-beta

[1.0.0-beta]: https://github.com/ahmetoner/whisper-asr-webservice/releases/tag/1.0.0-beta
