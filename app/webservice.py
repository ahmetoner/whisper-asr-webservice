import importlib.metadata
import io
import os
from os import path
from typing import Annotated, Optional, Union
from urllib.parse import quote

import aiohttp
import click
import torch
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications, Request, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

print(f"🚀 Starting Whisper ASR Service...")
print(f"Engine: {CONFIG.ASR_ENGINE}")
print(f"Model: {CONFIG.MODEL_NAME}")
print(f"Device: {CONFIG.DEVICE}")
print(f"Quantization: {CONFIG.MODEL_QUANTIZATION}")
print(f"Model Path: {CONFIG.MODEL_PATH}")

asr_model = ASRModelFactory.create_asr_model()
print("📥 Loading model...")
asr_model.load_model()
print("✅ Model loaded successfully!")

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


@app.on_event("startup")
async def startup_event():
    print("🔧 Running startup checks...")
    if asr_model.model is None:
        print("⚠️  Warning: Model not loaded, forcing load...")
        asr_model.load_model()
    else:
        print("✅ Model confirmed loaded and ready!")

    # Force GPU memory pre-allocation for consistent performance
    if CONFIG.DEVICE == "cuda" and torch.cuda.is_available():
        print("🚀 Pre-allocating GPU memory for consistent P99 performance...")
        try:
            # Allocate memory for maximum concurrent requests
            max_concurrent = asr_model.get_max_concurrent_requests()

            # Force PyTorch to allocate memory upfront
            torch.cuda.empty_cache()

            # Pre-allocate tensor memory based on expected workload
            estimated_memory_per_request = 2.0  # GB per request (conservative)
            total_memory_to_allocate = max_concurrent * estimated_memory_per_request * 0.8  # 80% to be safe

            # Create a large tensor to force memory allocation
            memory_bytes = int(total_memory_to_allocate * 1024**3 / 4)  # Divide by 4 for float32
            if memory_bytes > 0:
                dummy_tensor = torch.zeros(memory_bytes, dtype=torch.float32, device='cuda')
                del dummy_tensor  # Free but keep memory reserved

            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"✅ GPU memory pre-allocated - Reserved: {reserved:.2f}GB, Allocated: {allocated:.2f}GB")

        except Exception as e:
            print(f"⚠️  GPU memory pre-allocation failed (continuing anyway): {e}")

    print(f"🎯 Service ready on device: {CONFIG.DEVICE}")


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


# pre-request hook for authentication
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    EXCLUDED_PATHS = {"/", "/health", "/ready", "/docs", "/openapi.json", "/redoc"}
    if request.url.path in EXCLUDED_PATHS or not CONFIG.API_KEY:
        return await call_next(request)

    if request.headers.get("Authorization") != f"Bearer {CONFIG.API_KEY}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    return await call_next(request)


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["Health"])
async def ready():
    """Check if the service is ready to process requests (model loaded)"""
    if asr_model.model is None:
        return {"ready": False, "message": "Model not loaded"}, 503

    # Get GPU and concurrency info
    gpu_info = {}
    if hasattr(asr_model, 'get_max_concurrent_requests'):
        max_concurrent = asr_model.get_max_concurrent_requests()
        semaphore = asr_model.get_request_semaphore()
        current_available = semaphore._value
        current_active = max_concurrent - current_available

        gpu_info = {
            "max_concurrent_requests": max_concurrent,
            "active_requests": current_active,
            "available_slots": current_available,
        }

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_info["gpu_memory_gb"] = round(gpu_memory, 1)

    return {
        "ready": True,
        "engine": CONFIG.ASR_ENGINE,
        "model": CONFIG.MODEL_NAME,
        "device": CONFIG.DEVICE,
        "quantization": CONFIG.MODEL_QUANTIZATION,
        **gpu_info,
    }


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
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
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    audio_data = await load_audio(audio_file.file, encode)
    result = asr_model.transcribe(
        audio_data,
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {"diarize": diarize, "min_speakers": min_speakers, "max_speakers": max_speakers},
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


class SourceUriBody(BaseModel):
    source_uri: str
    prompt: Optional[str] = None


@app.post("/asr-source-uri", tags=["Endpoints"])
async def asr_source_uri(body: SourceUriBody):
    import asyncio

    # Configure timeout and connection settings
    timeout = aiohttp.ClientTimeout(
        total=300,  # 5 minutes total timeout
        connect=30,  # 30 seconds to establish connection
        sock_read=60,  # 60 seconds for reading data
    )

    connector = aiohttp.TCPConnector(
        limit=100,  # Connection pool limit
        limit_per_host=30,  # Connections per host
        keepalive_timeout=60,  # Keep connections alive
        enable_cleanup_closed=True,
    )

    try:
        async with aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": "Whisper-ASR-WebService/1.0",
                "Referer": "https://whisper.nectarsocial.com",
                "Origin": "https://whisper.nectarsocial.com",
            },
        ) as session:
            print(f"🌐 Downloading audio from: {body.source_uri}")

            # Add retry logic for failed downloads
            for attempt in range(3):  # 3 attempts
                try:
                    async with session.get(body.source_uri) as response:
                        if response.status != 200:
                            raise HTTPException(
                                status_code=400, detail=f"Failed to download audio: HTTP {response.status}"
                            )

                        # Check content type
                        content_type = response.headers.get('content-type', '')
                        if not any(
                            audio_type in content_type.lower()
                            for audio_type in ['audio/', 'video/', 'application/octet-stream']
                        ):
                            print(f"⚠️  Warning: Unexpected content type: {content_type}")

                        # Read audio data
                        audio_file_raw = await response.read()
                        print(f"✅ Downloaded {len(audio_file_raw)} bytes")
                        break

                except asyncio.TimeoutError:
                    if attempt == 2:  # Last attempt
                        raise HTTPException(status_code=408, detail=f"Timeout downloading audio from {body.source_uri}")
                    print(f"⏰ Timeout on attempt {attempt + 1}, retrying...")
                    await asyncio.sleep(2**attempt)  # Exponential backoff

                except Exception as e:
                    if attempt == 2:  # Last attempt
                        raise HTTPException(status_code=400, detail=f"Error downloading audio: {str(e)}")
                    print(f"❌ Error on attempt {attempt + 1}: {e}")
                    await asyncio.sleep(2**attempt)  # Exponential backoff

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

    audio_file = io.BytesIO(audio_file_raw)
    audio_data = await load_audio(audio_file, True)
    result = asr_model.transcribe(
        audio_data,
        "transcribe",
        None,
        body.prompt,
        True,
        True,
        None,
        "json",
    )

    return StreamingResponse(result, media_type="application/json", headers={"Asr-Engine": CONFIG.ASR_ENGINE})


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    audio_data = await load_audio(audio_file.file, encode)
    detected_lang_code, confidence = asr_model.language_detection(audio_data)
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
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start()
