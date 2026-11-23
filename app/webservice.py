import importlib.metadata
import logging
import os
import time
from contextlib import asynccontextmanager
from os import path
from threading import Thread
from typing import Annotated, Literal, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import Depends, FastAPI, File, Query, Request, Security, UploadFile
from fastapi.exception_handlers import http_exception_handler, request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from whisper import tokenizer

from app import openai_api, service
from app.config import CONFIG, CTRANSLATE2_GPU_UNSUPPORTED
from app.results import format_result

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
    "Settings: model path %s, beam size %s, batch size %s, VAD filter %s, idle timeout %ss, "
    "unload after request %s, API key %s",
    CONFIG.MODEL_PATH, CONFIG.BEAM_SIZE, CONFIG.BATCH_SIZE, CONFIG.VAD_FILTER, CONFIG.MODEL_IDLE_TIMEOUT,
    CONFIG.MODEL_UNLOAD_AFTER_REQUEST,
    "set" if CONFIG.API_KEY else "not set",
)
if CTRANSLATE2_GPU_UNSUPPORTED and not os.getenv("ASR_ENGINE"):
    logger.info("Using the openai_whisper engine, the only one that supports the GPU on this platform")
if CONFIG.ASR_ENGINE == "whisperx" and not CONFIG.HF_TOKEN:
    logger.warning("HF_TOKEN is not set, so WhisperX speaker diarization is disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    Thread(target=service.load_initial_model, daemon=True).start()
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
    openapi_tags=[
        {"name": "Audio", "description": "Transcribe or translate audio files"},
        {"name": "Models", "description": "List the available models"},
        {"name": "Legacy", "description": "The original endpoints, kept for existing integrations"},
    ],
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
    lifespan=lifespan,
    max_body_size=CONFIG.MAX_UPLOAD_SIZE_MB * 1024 * 1024 if CONFIG.MAX_UPLOAD_SIZE_MB > 0 else None,
    docs_url=None,
)

if CONFIG.CORS_ALLOW_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CONFIG.CORS_ALLOW_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["Content-Disposition", "Asr-Engine"],
    )

assets_path = os.getcwd() + "/swagger-ui-assets"
swagger_assets = path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js")
if swagger_assets:
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")


@app.get("/docs", include_in_schema=False)
async def swagger_ui(request: Request):
    root_path = request.scope.get("root_path", "")
    assets = {}
    if swagger_assets:
        assets = {
            "swagger_css_url": f"{root_path}/assets/swagger-ui.css",
            "swagger_js_url": f"{root_path}/assets/swagger-ui-bundle.js",
            "swagger_favicon_url": "",
        }
    return get_swagger_ui_html(
        openapi_url=f"{root_path}{app.openapi_url}",
        title=f"{app.title} - Swagger UI",
        swagger_ui_parameters=app.swagger_ui_parameters,
        **assets,
    )


OPENAI_ERROR_TYPES = {401: "invalid_request_error", 404: "invalid_request_error", 429: "rate_limit_error"}


def _is_openai_request(request: Request) -> bool:
    path = request.url.path
    root_path = request.scope.get("root_path", "")
    if root_path and path.startswith(root_path):
        path = path[len(root_path):]
    return path.startswith("/v1/")


def _openai_error_response(status_code: int, detail) -> JSONResponse:
    detail = detail if isinstance(detail, dict) else {"message": str(detail)}
    error_type = OPENAI_ERROR_TYPES.get(status_code, "server_error" if status_code >= 500 else "invalid_request_error")
    code = detail.get("code") or ("invalid_api_key" if status_code == 401 else None)
    error = {"message": detail["message"], "type": error_type, "param": detail.get("param"), "code": code}
    return JSONResponse({"error": error}, status_code=status_code)


@app.exception_handler(StarletteHTTPException)
async def handle_http_error(request: Request, exc: StarletteHTTPException):
    if _is_openai_request(request):
        return _openai_error_response(exc.status_code, exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    if _is_openai_request(request):
        error = exc.errors()[0]
        param = str(error["loc"][-1]).removesuffix("[]") if error.get("loc") else None
        return _openai_error_response(400, {"message": f"{param}: {error['msg']}", "param": param})
    return await request_validation_exception_handler(request, exc)


app.include_router(openai_api.router)


@app.get("/", include_in_schema=False)
async def index(request: Request):
    return RedirectResponse(f"{request.scope.get('root_path', '')}/docs")


@app.get("/health", include_in_schema=False)
async def health():
    if not service.model_ready.is_set():
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok"}


@app.post(
    "/asr", tags=["Legacy"], dependencies=[Security(service.verify_api_key), Depends(service.limit_pending_requests)]
)
def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Decode the file; set to false to send raw 16 kHz mono 16-bit PCM"),
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
        include_in_schema=CONFIG.ASR_ENGINE in ("faster_whisper", "openai_whisper"),
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
    audio = service.read_audio(audio_file, encode)
    result = service.asr_model.transcribe(
        audio,
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
    )
    logger.info(
        "Transcribed %r (%.1fs of audio, task %s, language %s, output %s) in %.1fs",
        audio_file.filename, len(audio) / CONFIG.SAMPLE_RATE, task, language or "auto", output, time.time() - start,
    )
    return Response(
        format_result(result, output),
        media_type=MEDIA_TYPES[output],
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post(
    "/detect-language",
    tags=["Legacy"],
    dependencies=[Security(service.verify_api_key), Depends(service.limit_pending_requests)],
)
def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Decode the file; set to false to send raw 16 kHz mono 16-bit PCM"),
    vad_filter: Annotated[
        bool,
        Query(
            description="Skip parts of the audio without speech before detecting the language",
            include_in_schema=CONFIG.ASR_ENGINE == "faster_whisper",
        ),
    ] = CONFIG.VAD_FILTER,
):
    start = time.time()
    audio = service.read_audio(audio_file, encode)
    detected_lang_code, confidence = service.asr_model.language_detection(audio, vad_filter)
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
    uvicorn.run(app, host=host, port=port, root_path=CONFIG.ROOT_PATH, log_level=CONFIG.LOG_LEVEL.lower(), log_config=None)


if __name__ == "__main__":
    start()
