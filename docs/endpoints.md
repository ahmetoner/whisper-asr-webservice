## Quick start

After running the docker image interactive Swagger API documentation is available at [localhost:9000/docs](http://localhost:9000/docs)

There are 2 endpoints available:

- /asr (TXT, VTT, SRT, TSV, JSON)
- /detect-language

## Automatic Speech recognition service /asr

If you choose the **transcribe** task, transcribes the uploaded file. Both audio and video files are supported (as long as ffmpeg supports it).

Note that you can also upload video formats directly as long as they are supported by ffmpeg.

You can get TXT, VTT, SRT, TSV and JSON output as a file from /asr endpoint.

You can provide the language or it will be automatically recognized.

If you choose the **translate** task it will provide an English transcript no matter which language was spoken.

You can enable word level timestamps output by `word_timestamps` parameter

You can Enable the voice activity detection (VAD) to filter out parts of the audio without speech  by `vad_filter` parameter (only with `Faster Whisper` for now).

Returns a json with following fields:

- **text**: Contains the full transcript
- **segments**: Contains an entry per segment. Each entry provides `timestamps`, `transcript`, `token ids`, `word level timestamps` and other metadata
- **language**: Detected or provided language (as a language code)

## Language detection service /detect-language

Detects the language spoken in the uploaded file. For longer files it only processes first 30 seconds.

Returns a json with following fields:

- **detected_language**
- **language_code**