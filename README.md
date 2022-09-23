# Whisper ASR Webservice

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification. For more details: [github.com/openai/whisper](https://github.com/openai/whisper/)

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

# Quick start

After running the docker image or ``poetry run whisper_asr`` interactive Swagger API documentation is available at [localhost:9000/docs](localhost:9000/docs)

Simply upload your sound file and choose either **translate** or **transcribe**. Optionally you can provide the language of the input file, otherwise it will be automatically detected.



## Build

Run

```sh
poetry build
```

### Configuring the Model

```sh
export ASR_MODEL=base
```

## Docker

### Build Image

```sh
docker build -t whisper-asr-webservice .
```

### Run Container

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