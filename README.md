![Release](https://img.shields.io/github/v/release/ahmetoner/whisper-asr-webservice.svg)
![Docker Pulls](https://img.shields.io/docker/pulls/onerahmet/openai-whisper-asr-webservice.svg)
![Build](https://img.shields.io/github/actions/workflow/status/ahmetoner/whisper-asr-webservice/docker-publish.yml.svg)
![Licence](https://img.shields.io/github/license/ahmetoner/whisper-asr-webservice.svg)
# Whisper ASR Webservice

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitask model that can perform multilingual speech recognition as well as speech translation and language identification. For more details: [github.com/openai/whisper](https://github.com/openai/whisper/)

## Features
Current release (v1.2.0) supports following whisper models:

- [openai/whisper](https://github.com/openai/whisper)@[v20230918](https://github.com/openai/whisper/releases/tag/v20230918)
- [guillaumekln/faster-whisper](https://github.com/guillaumekln/faster-whisper)@[0.9.0](https://github.com/guillaumekln/faster-whisper/releases/tag/v0.9.0)
- [whisperX](https://github.com/m-bain/whisperX)@[v3.1.1](https://github.com/m-bain/whisperX/releases/tag/v3.1.1)


## Quick Usage

### CPU

```sh
docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
```

### GPU

```sh
docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest-gpu
```

for more information:

- [Documentation/Run](https://ahmetoner.github.io/whisper-asr-webservice/run)
- [Docker Hub](https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice)

# Documentation
Explore the documentation by clicking [here](https://ahmetoner.github.io/whisper-asr-webservice).
