# API

The service exposes two endpoints. Both accept a file upload as `multipart/form-data` in the `audio_file` field.
Interactive documentation is available at `/docs`, and the OpenAPI schema at `/openapi.json`.

| Method | Path               | Description                                 |
|--------|--------------------|---------------------------------------------|
| `POST` | `/asr`             | Transcribe or translate an audio file       |
| `POST` | `/detect-language` | Detect the spoken language of an audio file |

## Transcription

`POST /asr`

Transcribes the uploaded file in its original language, or translates it into English.

### Parameters

| Name              | Type    | Default        | Description                                                                 |
|-------------------|---------|----------------|-----------------------------------------------------------------------------|
| `audio_file`      | file    | required       | Audio or video file                                                         |
| `task`            | string  | `transcribe`   | `transcribe` or `translate` (into English)                                  |
| `language`        | string  | auto-detected  | Language code of the audio, e.g. `en`, `de`, `tr`                           |
| `output`          | string  | `txt`          | `txt`, `json`, `vtt`, `srt` or `tsv`                                        |
| `initial_prompt`  | string  | none           | Text that guides the style or vocabulary of the transcript                  |
| `encode`          | boolean | `true`         | Decode the file. Set to `false` to send raw PCM (see below)                 |
| `word_timestamps` | boolean | `false`        | Include word-level timestamps. `openai_whisper` and `faster_whisper`        |
| `vad_filter`      | boolean | `false`        | Skip parts of the audio without speech. `faster_whisper` only               |
| `diarize`         | boolean | `false`        | Label segments by speaker. `whisperx` only, requires `HF_TOKEN`             |
| `min_speakers`    | integer | none           | Lower bound on the number of speakers. `whisperx` only                      |
| `max_speakers`    | integer | none           | Upper bound on the number of speakers. `whisperx` only                      |

All parameters except `audio_file` are query parameters. Some engine-specific parameters are hidden in the Swagger UI
when a different engine is configured.

### Examples

```shell
# Plain text transcript
curl -F "audio_file=@meeting.mp3" "http://localhost:9000/asr"

# SRT subtitles for a video
curl -F "audio_file=@video.mp4" "http://localhost:9000/asr?output=srt" -o video.srt

# English translation of German audio
curl -F "audio_file=@interview.wav" "http://localhost:9000/asr?task=translate&language=de"

# Speaker diarization with WhisperX
curl -F "audio_file=@call.m4a" "http://localhost:9000/asr?output=json&diarize=true&max_speakers=2"
```

### Output Formats

| `output` | Content type                 | Description                                         |
|----------|------------------------------|-----------------------------------------------------|
| `txt`    | `text/plain`                 | Plain transcript                                    |
| `json`   | `application/json`           | Full result with segments, timestamps and metadata  |
| `vtt`    | `text/vtt`                   | WebVTT subtitles                                    |
| `srt`    | `application/x-subrip`       | SubRip subtitles                                    |
| `tsv`    | `text/tab-separated-values`  | Segment start, end (in milliseconds) and text       |

The JSON output contains:

- `text`: the full transcript
- `segments`: one entry per segment, with start and end times, text and engine-specific metadata such as word
  timestamps and, when diarization is enabled, the speaker label
- `language`: the detected or requested language code

The response includes an `Asr-Engine` header with the engine that produced it.

### Input

With `encode=true` (the default), any audio or video format supported by FFmpeg is accepted, including MP3, WAV,
FLAC, OGG, Opus, M4A, MP4, MOV, MKV and WebM. The file is decoded in-process with PyAV and converted to 16 kHz mono
before transcription.

With `encode=false`, the file must be raw 16-bit little-endian mono PCM at 16 kHz. This skips decoding and is useful
when the client already has decoded audio.

## Language Detection

`POST /detect-language`

Detects the language spoken in the first 30 seconds of the file.

| Name         | Type    | Default  | Description                                   |
|--------------|---------|----------|-----------------------------------------------|
| `audio_file` | file    | required | Audio or video file                           |
| `encode`     | boolean | `true`   | Decode the file                               |

```shell
curl -F "audio_file=@audio.mp3" "http://localhost:9000/detect-language"
```

```json
{
  "detected_language": "english",
  "language_code": "en",
  "confidence": 0.98
}
```

## Health Check

The service starts accepting connections right away and loads the model in the background. Until the model is
loaded, `GET /health` returns `503` with `{"status": "loading"}`, and `/asr` and `/detect-language` requests wait for
the model. Afterwards it returns `{"status": "ok"}`. It does not require the API key, so container orchestrators and
load balancers can use it.

```shell
curl http://localhost:9000/health
```

## Authentication

When the `API_KEY` environment variable is set, both endpoints require the key in the `X-API-Key` header. The
documentation pages stay public, and the Swagger UI provides an **Authorize** button to enter the key.

```shell
curl -H "X-API-Key: your-secret-key" -F "audio_file=@audio.mp3" "http://localhost:9000/asr"
```

## Errors

| Status | Cause                                                                     |
|--------|---------------------------------------------------------------------------|
| `400`  | The file could not be decoded or contains no audio                        |
| `401`  | `API_KEY` is set and the `X-API-Key` header is missing or wrong           |
| `413`  | The upload is larger than `MAX_UPLOAD_SIZE_MB`                            |
| `422`  | A parameter has an invalid value, e.g. an unknown `output` or `language`  |
| `503`  | `MAX_PENDING_REQUESTS` requests are already being processed or waiting    |

Error responses are JSON objects with a `detail` field describing the problem, except `413`, which is plain text.
