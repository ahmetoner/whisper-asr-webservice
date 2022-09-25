# Whisper ASR Webservice

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification. For more details: [github.com/openai/whisper](https://github.com/openai/whisper/)

## Run (Docker Hub)
Whisper ASR Webservice now available on Docker Hub. You can find the latest version of this repository on https://hub.docker.com/r/onerahmet/openai-whisper-asr-webservice

```sh
docker pull onerahmet/openai-whisper-asr-webservice

docker run -d -p 9000:9000 onerahmet/openai-whisper-asr-webservice
# or
docker run -d -p 9000:9000 -e ASR_MODEL=base onerahmet/openai-whisper-asr-webservice

# Interactive Swagger API documentation is available at http://localhost:9000/docs

```
Available ASR_MODELs are `tiny`, `base`, `small`, `medium` and `large`

For English-only applications, the `.en` models tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.

## Run (Development Environment)

Enable venv:
```sh
python3.9 -m venv venv
source venv/bin/activate
```

Install poetry with following command:
```sh
pip3 install poetry==1.2.0
```

Install packages:
```sh
poetry install
```

Starting the Webservice:
```sh
poetry run whisper_asr
```

## Quick start

After running the docker image or `poetry run whisper_asr` interactive Swagger API documentation is available at [localhost:9000/docs](http://localhost:9000/docs)

There are two endpoints available: 
- /asr
-  /detect-language

## Automatic Speech recognition service /asr

If you choose the **transcribe** task, transcribes the uploaded sound file. 
You can provide the language or it will be automatically recognized. 
If you choose the **translate** task it will provide an English transcript no matter which language was spoken.

Returns a json with following fields:
- **text**: Contains the full transcript
- **segments**: Contains an entry per segment. Each entry  provides time stamps, transcript, token ids and other metadata
- **language**: Detected or provided language (as a language code)

## Language detection service /detect-language

Detects the language spoken in the uploaded sound file. For longer files it only processes first 30 seconds.

Returns a json with following fields:
- **detected_language**
- **langauge_code**

## Build

Build .whl package
```sh
poetry build
```

Configuring the Model
```sh
export ASR_MODEL=base
```

## Docker

Build Image
```sh
docker build -t whisper-asr-webservice .
```

Run Container
```sh
docker run -d -p 9000:9000 whisper-asr-webservice
# or
docker run -d -p 9000:9000 -e ASR_MODEL=base whisper-asr-webservice
```

## TODO

* Detailed README file
* Github pipeline
* Unit tests
* CUDA version of Docker image
* Hosted Swagger documentation with descriptions
* VTT and SRT output
* Recognize from path
* Batch recognition from given path/folder
* Live recognition support with HLS
* gRPC support
