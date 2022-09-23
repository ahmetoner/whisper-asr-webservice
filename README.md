# Whisper ASR Webservice

The webservice will be available soon.

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition as well as speech translation and language identification.

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
* Github ipeline
* Unit tests