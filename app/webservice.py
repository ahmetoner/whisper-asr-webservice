import importlib.metadata
import os
from os import path
from typing import Annotated, BinaryIO, Union, List
from urllib.parse import quote

import ffmpeg
import numpy as np
from fastapi import FastAPI, File, Query, UploadFile, applications, Form, Request
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")
if ASR_ENGINE == "faster_whisper":
    from .faster_whisper.core import language_detection, transcribe
else:
    from .openai_whisper.core import language_detection, transcribe

SAMPLE_RATE = 16000
LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": projectMetadata["License"]},
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


@app.post("/v1/audio/transcriptions", tags=["Endpoints"])
@app.post("/v1/audio/translations", tags=["Endpoints"])
async def asr_openai(
    request: Request,
    file: UploadFile = File(...),
    # Required by OpenAI API but unused for now
    model: str = Form(enum=["whisper-1"]),
    language: Union[str, None] = Form(default="", enum=LANGUAGE_CODES),
    prompt: Union[str, None] = Form(default=""),
    response_format: Union[str, None] = Form(default="json", enum=["json", "text", "srt", "verbose_json", "vtt"]),
    temperature: Union[float, None] = Form(default=0),
    timestamp_granularities: List[str] = Form(default=["segment"], alias="timestamp_granularities[]", description="Word-level timestamps on translations may not be reliable.")
):

    result = transcribe(
        load_audio(file.file, True),
        "transcribe" if request.url.path == "/v1/audio/transcriptions" else "translate",
        language,
        prompt,
        True if ASR_ENGINE == "faster_whisper" else False,
        "word" in timestamp_granularities,
        timestamp_granularities,
        temperature,
        response_format
    )

    c_type = {
        "json": "application/json",
        "verbose_json": "application/json",
        "text": "text/plain",
        "srt": "text/plain",
        "vtt": "text/plain"
    }

    return StreamingResponse(
        result,
        media_type=c_type[response_format],
        headers={
            "Asr-Engine": ASR_ENGINE
        },
    )


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
