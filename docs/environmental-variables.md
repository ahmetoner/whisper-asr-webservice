# Configuration

The service is configured with environment variables. All of them are optional.

| Variable                   | Default                         | Description                                                    |
|----------------------------|---------------------------------|----------------------------------------------------------------|
| `ASR_ENGINE`               | `faster_whisper`                | `faster_whisper`, `openai_whisper` or `whisperx`; `openai_whisper` by default on ARM64 GPUs |
| `ASR_MODEL`                | `base`                          | Model to load, see [Models](#models)                           |
| `ASR_MODEL_PATH`           | `$XDG_CACHE_HOME/whisper`       | Model download directory (`openai_whisper`, `faster_whisper`)  |
| `XDG_CACHE_HOME`           | `~/.cache`                      | Base directory for downloaded models and data                  |
| `ASR_DEVICE`               | `cuda` if available, else `cpu` | Device to run the model on                                     |
| `ASR_CPU_THREADS`          | engine default                  | CPU threads for transcription, see [Device and Quantization](#device-and-quantization) |
| `ASR_QUANTIZATION`         | `float32` on GPU, `int8` on CPU | Weight precision (`faster_whisper`, `whisperx`)                |
| `ASR_BEAM_SIZE`            | engine default                  | Beam size used for decoding                                    |
| `ASR_BATCH_SIZE`           | engine default                  | Batch size for batched inference (`faster_whisper`, `whisperx`) |
| `ASR_VAD_FILTER`           | `false`                         | Default value of the `vad_filter` request parameter            |
| `ASR_VAD_THRESHOLD`        | engine default                  | Speech probability threshold of the VAD, see [Voice Activity Detection](#voice-activity-detection) |
| `ASR_VAD_MIN_SILENCE_MS`   | engine default                  | Silence length that ends a speech segment (`faster_whisper`)   |
| `ASR_VAD_SPEECH_PAD_MS`    | engine default                  | Padding added around speech segments (`faster_whisper`)        |
| `ASR_CONDITION_ON_PREVIOUS_TEXT` | engine default            | Use the previous text as context for the next window, see [Decoding](#decoding) |
| `ASR_PROMPT_RESET_ON_TEMPERATURE` | engine default           | Reset the prompt above this fallback temperature (`faster_whisper`, `whisperx`) |
| `MODEL_IDLE_TIMEOUT`       | `0`                             | Seconds of inactivity before the model is unloaded; `0` keeps it loaded |
| `MODEL_UNLOAD_AFTER_REQUEST` | `false`                       | Unload the model after every request, see [Model Unloading](#model-unloading) |
| `API_KEY`                  | unset                           | Require this key in the `X-API-Key` request header             |
| `HF_TOKEN`                 | unset                           | Hugging Face token for WhisperX diarization                    |
| `SUBTITLE_MAX_LINE_WIDTH`  | unset                           | Maximum characters per subtitle line, see [Subtitles](#subtitles) |
| `SUBTITLE_MAX_LINE_COUNT`  | unset                           | Maximum lines per subtitle, see [Subtitles](#subtitles)        |
| `SUBTITLE_HIGHLIGHT_WORDS` | `false`                         | Underline the current word in subtitles, see [Subtitles](#subtitles) |
| `SAMPLE_RATE`              | `16000`                         | Sample rate audio is converted to. Whisper expects 16 kHz      |
| `MAX_UPLOAD_SIZE_MB`       | `0`                             | Largest accepted upload in MB; `0` means no limit, see [Limits](#limits) |
| `MAX_PENDING_REQUESTS`     | `0`                             | Requests processed or waiting at once; `0` means no limit      |
| `CORS_ALLOW_ORIGINS`       | unset                           | Comma-separated origins allowed to call the API from a browser |
| `ROOT_PATH`                | unset                           | URL prefix when served behind a reverse proxy, e.g. `/whisper` |
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

On a CPU, `faster_whisper` and `whisperx` use 4 threads by default. `ASR_CPU_THREADS` changes this, but more threads
are not always faster, for example on CPUs that mix performance and efficiency cores, so measure with your own audio
before changing it. On a CPU, `ASR_BATCH_SIZE` usually speeds up long files more than extra threads.

## Decoding

`ASR_BEAM_SIZE` and `ASR_BATCH_SIZE` tune decoding. When they are not set, each engine uses its own default.

With `faster_whisper`, setting `ASR_BATCH_SIZE` (for example `8`) enables batched inference, which transcribes long
files considerably faster on both GPU and CPU. Batched inference always applies voice activity detection to split the
audio and does not use the previous text as context.

Whisper transcribes in 30-second windows and passes the text of the previous window to the next one as context. This
keeps the style consistent, but it can also make the model stop too early, for example skipping speech at the end
of a file after the main content, or lose punctuation and capitalization over time. `ASR_CONDITION_ON_PREVIOUS_TEXT=false`
turns the context off (WhisperX already does this by default). A softer option is lowering
`ASR_PROMPT_RESET_ON_TEMPERATURE` (default `0.5`), for example to `0.3`, which resets the context only when decoding
becomes uncertain.

## Voice Activity Detection

Voice activity detection (VAD) skips the parts of the audio without speech. `ASR_VAD_FILTER=true` enables it by default
for `faster_whisper`; clients can still override it per request with the `vad_filter` parameter, which
`/detect-language` also accepts so the language is detected from speech instead of leading silence or music.

The VAD settings tune where speech starts and ends:

- `ASR_VAD_THRESHOLD`: speech probability above which audio counts as speech (default `0.5`). Lower it if quiet
  speech is skipped. With `whisperx`, it sets both the onset and offset of WhisperX's VAD, which is always on; values
  around `0.1` help when WhisperX drops the start or end of non-English audio.
- `ASR_VAD_MIN_SILENCE_MS`: how long a silence must be to end a speech segment.
- `ASR_VAD_SPEECH_PAD_MS`: padding added around each speech segment.

OpenAI API clients can set the same values per request with `chunking_strategy`, see
[OpenAI API](openai.md#request-fields).

## Model Unloading

When `MODEL_IDLE_TIMEOUT` is greater than `0`, the model is unloaded after that many seconds without requests, which
frees CPU or GPU memory. It is loaded again on the next request, which then takes longer to complete.

With `MODEL_UNLOAD_AFTER_REQUEST=true`, the model is unloaded as soon as each request finishes, so it only uses memory
while a request is processed. Every request then includes the model loading time. This suits occasional use on a
machine that needs its memory for other work.

## Subtitles

By default, the `srt` and `vtt` outputs contain one subtitle per segment. Set both `SUBTITLE_MAX_LINE_WIDTH` and
`SUBTITLE_MAX_LINE_COUNT` to split them into lines of at most that many characters instead, and
`SUBTITLE_HIGHLIGHT_WORDS=true` to underline each word as it is spoken. Both need word timestamps, so enable
`word_timestamps` or use the WhisperX engine, which always provides them.

## Limits

The model processes one request at a time, and every request waiting for it keeps its decoded audio in memory (about
230 MB per hour of audio). On a shared or public server, set limits so that load cannot exhaust the memory:

```shell
export MAX_UPLOAD_SIZE_MB=500       # larger uploads are rejected with 413 while they are being received
export MAX_PENDING_REQUESTS=4       # further requests are rejected with 503 until one finishes
```

Both limits are off by default.

## Browser Clients and Reverse Proxies

Browser-based applications can only call the API from another origin when that origin is allowed. List the origins
in `CORS_ALLOW_ORIGINS`, for example `CORS_ALLOW_ORIGINS=https://app.example.com,http://localhost:3000`, or `*` to
allow every origin.

When a reverse proxy serves the service under a path prefix and removes the prefix before forwarding requests, set
`ROOT_PATH` to that prefix (e.g. `ROOT_PATH=/whisper`) so that redirects, the documentation and the Swagger UI use
the right URLs.

## Logging

At `INFO`, the service logs its configuration at startup, model loading and unloading, and one line per request with
the file name, audio length and processing time. Set `LOG_LEVEL=DEBUG` to also log all settings, audio decoding
times, the options passed to the engine and the detected language, plus the health check requests that are hidden
otherwise.

```shell
docker run -d -p 9000:9000 -e LOG_LEVEL=DEBUG onerahmet/openai-whisper-asr-webservice:latest
```

## Authentication

When `API_KEY` is set, all API endpoints return `401 Unauthorized` unless the request includes the key, either as
`Authorization: Bearer <key>` or in the `X-API-Key` header. The documentation pages and `/health` remain public.

## Speaker Diarization

Diarization is available with the `whisperx` engine and requires a Hugging Face access token:

1. Create a token at <https://huggingface.co/settings/tokens>.
2. Accept the user conditions of
   [`pyannote/speaker-diarization-community-1`](https://huggingface.co/pyannote/speaker-diarization-community-1) with
   the same account.
3. Start the service with `ASR_ENGINE=whisperx` and `HF_TOKEN=<your token>`.

The `diarize`, `min_speakers` and `max_speakers` request parameters then become available.
