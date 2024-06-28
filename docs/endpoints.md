## Quick start

After running the docker image interactive Swagger API documentation is available at [localhost:9000/docs](http://localhost:9000/docs)

There are 2 endpoints available:

- [/asr](##Automatic-Speech-recognition-service-/asr) (Automatic Speech Recognition)
- [/detect-language](##Language-detection-service-/detect-language)

## Automatic speech recognition service /asr

- 2 task choices:
  - **transcribe**: (default) task, transcribes the uploaded file.
  - **translate**: will provide an English transcript no matter which language was spoken.
- Files are automatically converted with FFmpeg.
  - Full list of supported [audio](https://ffmpeg.org/general.html#Audio-Codecs) and [video](https://ffmpeg.org/general.html#Video-Codecs) formats.
- You can enable word level timestamps output by `word_timestamps` parameter
- You can Enable the voice activity detection (VAD) to filter out parts of the audio without speech  by `vad_filter` parameter (only with `Faster Whisper` for now).

### Request URL Query Params

| Name            | Values                                         |
|-----------------|------------------------------------------------|
| audio_file      | File                                           |
| output          | `text` (default), `json`, `vtt`, `strt`, `tsv` |
| task            | `transcribe`, `translate`                      |
| language        | `en` (default is auto recognition)             |
| word_timestamps | false (default)                                |
| encode          | true (default)                                 |

Example request with cURL

```bash
curl -X POST -H "content-type: multipart/form-data" -F "audio_file=@/path/to/file" 0.0.0.0:9000/asr?output=json
```

### Response (JSON)

- **text**: Contains the full transcript
- **segments**: Contains an entry per segment. Each entry provides `timestamps`, `transcript`, `token ids`, `word level timestamps` and other metadata
- **language**: Detected or provided language (as a language code)

## Language detection service /detect-language

Detects the language spoken in the uploaded file. Only processes first 30 seconds.

Returns a json with following fields:

- **detected_language**: "english"
- **language_code**: "en"
