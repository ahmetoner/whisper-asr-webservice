# Configuration

The service is configured with environment variables. All of them are optional.

| Variable                   | Default                         | Description                                                    |
|----------------------------|---------------------------------|----------------------------------------------------------------|
| `ASR_ENGINE`               | `openai_whisper`                | `openai_whisper`, `faster_whisper` or `whisperx`               |
| `ASR_MODEL`                | `base`                          | Model to load, see [Models](#models)                           |
| `ASR_MODEL_PATH`           | `$XDG_CACHE_HOME/whisper`       | Model download directory (`openai_whisper`, `faster_whisper`)  |
| `XDG_CACHE_HOME`           | `~/.cache`                      | Base directory for downloaded models and data                  |
| `ASR_DEVICE`               | `cuda` if available, else `cpu` | Device to run the model on                                     |
| `ASR_QUANTIZATION`         | `float32` on GPU, `int8` on CPU | Weight precision (`faster_whisper`, `whisperx`)                |
| `ASR_BEAM_SIZE`            | engine default                  | Beam size used for decoding                                    |
| `ASR_BATCH_SIZE`           | engine default                  | Batch size for batched inference (`faster_whisper`, `whisperx`) |
| `ASR_VAD_FILTER`           | `false`                         | Default value of the `vad_filter` request parameter            |
| `MODEL_IDLE_TIMEOUT`       | `0`                             | Seconds of inactivity before the model is unloaded; `0` keeps it loaded |
| `MODEL_UNLOAD_AFTER_REQUEST` | `false`                       | Unload the model after every request, see [Model Unloading](#model-unloading) |
| `API_KEY`                  | unset                           | Require this key in the `X-API-Key` request header             |
| `HF_TOKEN`                 | unset                           | Hugging Face token for WhisperX diarization                    |
| `SUBTITLE_MAX_LINE_WIDTH`  | `1000`                          | Maximum characters per subtitle line (`whisperx`)              |
| `SUBTITLE_MAX_LINE_COUNT`  | `2`                             | Maximum lines per subtitle (`whisperx`)                        |
| `SUBTITLE_HIGHLIGHT_WORDS` | `false`                         | Highlight the current word in subtitles (`whisperx`)           |
| `SAMPLE_RATE`              | `16000`                         | Sample rate audio is converted to. Whisper expects 16 kHz      |
| `MAX_UPLOAD_SIZE_MB`       | `0`                             | Largest accepted upload in MB; `0` means no limit, see [Limits](#limits) |
| `MAX_PENDING_REQUESTS`     | `0`                             | Requests processed or waiting at once; `0` means no limit      |
| `LOG_LEVEL`                | `INFO`                          | `DEBUG`, `INFO`, `WARNING` or `ERROR`, see [Logging](#logging) |

## Models

| Type              | Models                                                                                          |
|-------------------|-------------------------------------------------------------------------------------------------|
| Multilingual      | `tiny`, `base`, `small`, `medium`, `large-v1`, `large-v2`, `large-v3` (`large`), `large-v3-turbo` (`turbo`) |
| English only      | `tiny.en`, `base.en`, `small.en`, `medium.en`                                                   |
| Distilled         | `distil-small.en`, `distil-medium.en`, `distil-large-v2`, `distil-large-v3` (`faster_whisper`, `whisperx`) |

Larger models are more accurate but slower and need more memory. `turbo` is close to `large-v3` in accuracy at a
fraction of the cost and is a good default on GPU. For English-only audio, the `.en` models are more accurate than
their multilingual counterparts, especially at the `tiny` and `base` sizes. Distilled models trade a little accuracy
for faster inference.

## Device and Quantization

`ASR_QUANTIZATION` sets the precision of the model weights for `faster_whisper` and `whisperx`:

| Value     | Speed   | Accuracy | Notes                     |
|-----------|---------|----------|---------------------------|
| `float32` | slowest | highest  | Default on GPU            |
| `float16` | faster  | high     | Recommended on GPU        |
| `int8`    | fastest | lower    | Default on CPU            |

## Decoding

`ASR_BEAM_SIZE` and `ASR_BATCH_SIZE` tune decoding. When they are not set, each engine uses its own default.

With `faster_whisper`, setting `ASR_BATCH_SIZE` enables batched inference, which is significantly faster on GPU.
Batched inference always applies voice activity detection to split the audio.

`ASR_VAD_FILTER=true` enables voice activity detection by default for `faster_whisper`. Clients can still override it
per request with the `vad_filter` parameter.

## Model Unloading

When `MODEL_IDLE_TIMEOUT` is greater than `0`, the model is unloaded after that many seconds without requests, which
frees CPU or GPU memory. It is loaded again on the next request, which then takes longer to complete.

With `MODEL_UNLOAD_AFTER_REQUEST=true`, the model is unloaded as soon as each request finishes, so it only uses memory
while a request is processed. Every request then includes the model loading time. This suits occasional use on a
machine that needs its memory for other work.

## Limits

The model processes one request at a time, and every request waiting for it keeps its decoded audio in memory (about
230 MB per hour of audio). On a shared or public server, set limits so that load cannot exhaust the memory:

```shell
export MAX_UPLOAD_SIZE_MB=500       # larger uploads are rejected with 413 while they are being received
export MAX_PENDING_REQUESTS=4       # further requests are rejected with 503 until one finishes
```

Both limits are off by default.

## Logging

At `INFO`, the service logs its configuration at startup, model loading and unloading, and one line per request with
the file name, audio length and processing time. Set `LOG_LEVEL=DEBUG` to also log all settings, FFmpeg decoding
times, the options passed to the engine and the detected language, plus the health check requests that are hidden
otherwise.

```shell
docker run -d -p 9000:9000 -e LOG_LEVEL=DEBUG onerahmet/openai-whisper-asr-webservice:latest
```

## Authentication

When `API_KEY` is set, `/asr` and `/detect-language` return `401 Unauthorized` unless the request includes the key in
the `X-API-Key` header. The documentation pages remain public. See [API](endpoints.md#authentication).

## Speaker Diarization

Diarization is available with the `whisperx` engine and requires a Hugging Face access token:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the user conditions of
   [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) with
   the same account.
3. Start the service with `ASR_ENGINE=whisperx` and `HF_TOKEN=<your token>`.

The `diarize`, `min_speakers` and `max_speakers` request parameters then become available.
