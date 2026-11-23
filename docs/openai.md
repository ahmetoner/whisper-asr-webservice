# OpenAI-Compatible API

The service implements OpenAI's speech-to-text API under `/v1`, so applications and libraries built for OpenAI can use it
by changing the base URL to `http://localhost:9000/v1`.

| Method | Path                       | Description                                     |
|--------|----------------------------|-------------------------------------------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe an audio file                        |
| `POST` | `/v1/audio/translations`   | Translate an audio file into English            |
| `GET`  | `/v1/models`               | List the available models                       |
| `GET`  | `/v1/models/{model}`       | Retrieve a model                                |

## Examples

=== "Python"

    ```python
    from openai import OpenAI

    client = OpenAI(base_url="http://localhost:9000/v1", api_key="your-api-key")

    with open("audio.mp3", "rb") as audio:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=audio)
    print(transcript.text)
    ```

=== "JavaScript"

    ```javascript
    import fs from "fs";
    import OpenAI from "openai";

    const client = new OpenAI({ baseURL: "http://localhost:9000/v1", apiKey: "your-api-key" });

    const transcript = await client.audio.transcriptions.create({
      model: "whisper-1",
      file: fs.createReadStream("audio.mp3"),
    });
    console.log(transcript.text);
    ```

=== "curl"

    ```shell
    curl http://localhost:9000/v1/audio/transcriptions \
      -H "Authorization: Bearer your-api-key" \
      -F model=whisper-1 \
      -F file=@audio.mp3 \
      -F response_format=verbose_json \
      -F "timestamp_granularities[]=word"
    ```

When `API_KEY` is not set, any API key is accepted; the OpenAI libraries require one, so pass any value.

## Request Fields

| Field                        | Support                                                                                      |
|------------------------------|----------------------------------------------------------------------------------------------|
| `file`                       | Any audio or video format FFmpeg can decode                                                   |
| `model`                      | Required, but any value is accepted: the request is served by the model set with `ASR_MODEL` |
| `language`                   | ISO-639-1 code such as `en`; region suffixes (`en-US`) and names (`english`) also work       |
| `languages[]`                | One language is used like `language`; several fall back to automatic detection               |
| `prompt`, `keywords[]`       | Passed to the model as initial prompt; keywords are appended to the prompt                   |
| `response_format`            | `json` (default), `text`, `srt`, `vtt`, `verbose_json`, `diarized_json`                      |
| `timestamp_granularities[]`  | `segment` (default) and `word`, with `response_format=verbose_json`                           |
| `temperature`                | Sampling temperature between 0 and 1                                                         |
| `stream`                     | Server-sent events, see [Streaming](#streaming)                                              |
| `chunking_strategy`          | `auto` or a `server_vad` object enables voice activity detection (`faster_whisper`); its `threshold`, `silence_duration_ms` and `prefix_padding_ms` set the [VAD settings](environmental-variables.md#voice-activity-detection) for the request |
| `include[]`, `known_speaker_names[]`, `known_speaker_references[]` | Accepted and ignored                                  |

`/v1/audio/translations` accepts `file`, `model`, `prompt`, `response_format` (without `diarized_json`) and
`temperature`, and always returns English text.

## Responses

- `json`: `{"text": "...", "usage": {"type": "duration", "seconds": 11}}`
- `text`, `srt`, `vtt`: plain text
- `verbose_json`: `task`, `language`, `duration`, `text`, `segments`, `words` (with the `word` granularity) and `usage`,
  the same structure as the [`/asr` JSON output](endpoints.md#output-formats)
- `diarized_json`: segments labelled with speakers `A`, `B`, ... Requires `ASR_ENGINE=whisperx` and `HF_TOKEN`, see
  [Speaker Diarization](environmental-variables.md#speaker-diarization)

## Streaming

With `stream=true`, the response is a stream of server-sent events: a `transcript.text.delta` event per segment,
`transcript.text.segment` events with speakers for `diarized_json`, and a final `transcript.text.done` event with the
full text. With `faster_whisper`, segments are sent as soon as they are transcribed; the other engines send them when
the whole file is done.

```python
stream = client.audio.transcriptions.create(model="whisper-1", file=audio, stream=True)
for event in stream:
    if event.type == "transcript.text.delta":
        print(event.delta, end="", flush=True)
```

## Errors and Authentication

Errors use OpenAI's format, `{"error": {"message": ..., "type": ..., "param": ..., "code": ...}}`, so the OpenAI
libraries raise their usual exceptions. When `API_KEY` is set, send it as `Authorization: Bearer <key>`; the
`X-API-Key` header works as well.

## Applications

Any application that lets you set an OpenAI base URL for speech-to-text can use the service. For example:

- **Open WebUI**: in *Admin Settings → Audio*, choose the `OpenAI` speech-to-text engine, set the API base URL to
  `http://<host>:9000/v1` and the model to `whisper-1`.
- **LiteLLM**: add a model with `model: openai/whisper-1` and `api_base: http://<host>:9000/v1`.
