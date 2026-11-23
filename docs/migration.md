# Upgrading to v2

Version 2 adds an [OpenAI-compatible API](openai.md) and keeps the original `/asr` and `/detect-language` endpoints,
their parameters, the environment variables and the `X-API-Key` header. Most setups keep working unchanged; check the
points below.

## Default Engine

The default engine is now `faster_whisper`, which is several times faster than `openai_whisper` with the same
models. Transcripts can differ slightly in wording and punctuation. To keep the previous engine, set it explicitly:

```shell
-e ASR_ENGINE=openai_whisper
```

## JSON Output of `/asr`

`/asr?output=json` now returns the same structure for every engine, following OpenAI's `verbose_json` format. The
`txt`, `vtt`, `srt` and `tsv` outputs and `/detect-language` are unchanged.

| Before                                                     | Now                                                         |
|------------------------------------------------------------|-------------------------------------------------------------|
| `language` as a code, e.g. `en`                            | `language` as a name, e.g. `english`                        |
| Segments as arrays with `faster_whisper`                   | Segments as objects with named fields for every engine      |
| `word_segments` and `score` with `whisperx`                | `words` with `word`, `start`, `end` and `probability`       |
| Words inside each segment                                  | Words in a top-level `words` list                           |
| —                                                          | `task` and `duration`                                       |

See [Output Formats](endpoints.md#output-formats) for the full structure.

## Subtitles

`srt` and `vtt` outputs contain one subtitle per segment by default with every engine. Before, WhisperX merged
segments into longer subtitles. To split subtitles into lines, set `SUBTITLE_MAX_LINE_WIDTH` and
`SUBTITLE_MAX_LINE_COUNT`, see [Subtitles](environmental-variables.md#subtitles).

## FFmpeg

Audio is now decoded inside the service with PyAV, so the Docker images no longer contain the `ffmpeg` command and
running from source no longer needs FFmpeg installed. Supported formats are unchanged.

## Running from Source

The project uses [uv](https://docs.astral.sh/uv/) instead of Poetry. `uv sync` installs the CPU build of PyTorch and
the development tools; for NVIDIA GPUs use `uv sync --no-group cpu --group cuda` and then `uv run --no-sync` so
that the CUDA build is kept. Run commands with `uv run`, e.g. `uv run whisper-asr-webservice`.

## New Integrations

New integrations should use the [OpenAI-compatible API](openai.md): it works with the OpenAI SDKs and with
applications that let you set the OpenAI base URL.
