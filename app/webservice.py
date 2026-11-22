import importlib.metadata
import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
from os import path
from threading import Event, Thread
from typing import Annotated, Literal, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, Security, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

logging.basicConfig(
    level=max(logging.getLevelName(CONFIG.LOG_LEVEL), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("app").setLevel(CONFIG.LOG_LEVEL)
logger = logging.getLogger(__name__)
if CONFIG.LOG_LEVEL != "DEBUG":
    logging.getLogger("uvicorn.access").addFilter(lambda record: "/health" not in record.getMessage())


def _rename_uvicorn_logger(record):
    record.name = "uvicorn"
    return True


logging.getLogger("uvicorn.error").addFilter(_rename_uvicorn_logger)

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
logger.info(
    "Whisper ASR Webservice %s: engine %s, model %s, device %s, quantization %s",
    projectMetadata["Version"], CONFIG.ASR_ENGINE, CONFIG.MODEL_NAME, CONFIG.DEVICE, CONFIG.MODEL_QUANTIZATION,
)
logger.debug(
    "Settings: model path %s, beam size %s, batch size %s, VAD filter %s, idle timeout %ss, API key %s",
    CONFIG.MODEL_PATH, CONFIG.BEAM_SIZE, CONFIG.BATCH_SIZE, CONFIG.VAD_FILTER, CONFIG.MODEL_IDLE_TIMEOUT,
    "set" if CONFIG.API_KEY else "not set",
)
if CONFIG.ASR_ENGINE == "whisperx" and not CONFIG.HF_TOKEN:
    logger.warning("HF_TOKEN is not set, so WhisperX speaker diarization is disabled")

asr_model = ASRModelFactory.create_asr_model()
model_ready = Event()


def load_initial_model():
    try:
        with asr_model.loaded_model():
            pass
    except Exception:
        logger.exception("Failed to load the model")
        os._exit(1)
    model_ready.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=load_initial_model, daemon=True).start()
    yield


LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())
LanguageCode = Literal[tuple(LANGUAGE_CODES)]
MEDIA_TYPES = {
    "txt": "text/plain",
    "vtt": "text/vtt",
    "srt": "application/x-subrip",
    "tsv": "text/tab-separated-values",
    "json": "application/json",
}

projectUrls = dict(url.split(", ", 1) for url in projectMetadata.get_all("Project-URL") or [])
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectUrls.get("Homepage")},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
    lifespan=lifespan,
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


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: Annotated[Optional[str], Security(api_key_header)]):
    if CONFIG.API_KEY and not (api_key and secrets.compare_digest(api_key, CONFIG.API_KEY)):
        logger.warning("Rejected a request with a missing or invalid API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def read_audio(audio_file: UploadFile, encode: bool):
    try:
        return load_audio(audio_file.file, encode)
    except ValueError as e:
        logger.warning("Could not read %r: %s", audio_file.filename, e)
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.get("/health", include_in_schema=False)
async def health():
    if not model_ready.is_set():
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post("/asr", tags=["Endpoints"], dependencies=[Security(verify_api_key)])
def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Literal["transcribe", "translate"] = Query(default="transcribe"),
    language: Annotated[Optional[LanguageCode], Query()] = None,
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = CONFIG.VAD_FILTER,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    output: Literal["txt", "vtt", "srt", "tsv", "json"] = Query(default="txt"),
):
    start = time.time()
    audio = read_audio(audio_file, encode)
    result = asr_model.transcribe(
        audio,
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
        output,
    )
    logger.info(
        "Transcribed %r (%.1fs of audio, task %s, language %s, output %s) in %.1fs",
        audio_file.filename, len(audio) / CONFIG.SAMPLE_RATE, task, language or "auto", output, time.time() - start,
    )
    return StreamingResponse(
        result,
        media_type=MEDIA_TYPES[output],
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"], dependencies=[Security(verify_api_key)])
def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    start = time.time()
    detected_lang_code, confidence = asr_model.language_detection(read_audio(audio_file, encode))
    logger.info(
        "Detected language %s (%.2f) in %r in %.1fs", detected_lang_code, confidence, audio_file.filename, time.time() - start
    )
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None):
    uvicorn.run(app, host=host, port=port, log_level=CONFIG.LOG_LEVEL.lower(), log_config=None)


if __name__ == "__main__":
    start()
