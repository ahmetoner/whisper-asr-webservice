import importlib.metadata
import os
from os import path
from typing import BinaryIO, Union

import ffmpeg
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")
HF_TOKEN = os.getenv("HF_TOKEN", "")

if ASR_ENGINE == "faster_whisper":
    from .faster_whisper.core import transcribe, language_detection
elif ASR_ENGINE == "whisperx":
    from .mbain_whisperx.core import transcribe, language_detection
else:
    from .openai_whisper.core import transcribe, language_detection

SAMPLE_RATE = 16000
LANGUAGE_CODES = sorted(list(tokenizer.LANGUAGES.keys()))

projectMetadata = importlib.metadata.metadata('whisper-asr-webservice')
app = FastAPI(
    title=projectMetadata['Name'].title().replace('-', ' '),
    description=projectMetadata['Summary'],
    version=projectMetadata['Version'],
    contact={
        "url": projectMetadata['Home-page']
    },
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={
        "name": "MIT License",
        "url": projectMetadata['License']
    }
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")


    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )


    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
def asr(
    task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    audio_file: UploadFile = File(...),
    encode : bool = Query(default=True, description="Encode audio first through ffmpeg"),
    output : Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
    word_timestamps : bool = Query(
        default=False, 
        description="World level timestamps", 
        include_in_schema=(True if ASR_ENGINE == "faster_whisper" else False)
    ),
    diarize : bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if ASR_ENGINE == "whisperx" and HF_TOKEN != "" else False)),
    min_speakers : Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if ASR_ENGINE == "whisperx" else False)),
    max_speakers : Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if ASR_ENGINE == "whisperx" else False)),
):
    result = transcribe(
        load_audio(audio_file.file, encode),
        task, 
        language,
        initial_prompt,
        word_timestamps,
        {
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers
        },
        output)
    
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            'Asr-Engine': ASR_ENGINE,
            'Content-Disposition': f'attachment; filename="{audio_file.filename}.{output}"'
        })


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg")
):
    detected_lang_code = language_detection(load_audio(audio_file.file, encode))
    return {"detected_language": tokenizer.LANGUAGES[detected_lang_code], "language_code": detected_lang_code}


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
