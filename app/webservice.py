import asyncio
import io
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from os import path
from threading import Lock
from contextlib import suppress
from typing import Annotated, Dict, List, Literal, Optional, Union
from urllib.parse import quote
from uuid import uuid4

import click
import uvicorn
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

asr_model = ASRModelFactory.create_asr_model()
asr_model.load_model()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())


def _parse_positive_int_env(key: str, default: int) -> int:
    value = os.getenv(key)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(1, parsed)


def _parse_non_negative_float_env(key: str, default: float) -> float:
    value = os.getenv(key)
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return max(0.0, parsed)


ASYNC_ASR_WORKER_COUNT = _parse_positive_int_env("ASYNC_ASR_WORKER_COUNT", 3) # 并发处理任务的线程数
ASYNC_ASR_JOB_TIMEOUT = _parse_non_negative_float_env("ASYNC_ASR_JOB_TIMEOUT", 180) # 任务超时时间，单位：秒


class AsyncASRRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    file_url: HttpUrl = Field(..., alias="fileUrl", description="远程音频文件的 URL。")
    callback_url: HttpUrl = Field(..., alias="callbackUrl", description="识别完成后异步回调地址。")
    encode: bool = Field(
        True,
        description="是否先通过 FFmpeg 转码再送入模型。",
    )
    task: Literal["transcribe", "translate"] = Field(
        "transcribe",
        description="识别任务类型。",
    )
    language: Optional[str] = Field(
        None,
        description="指定语言代码（留空则自动识别）。",
    )
    initial_prompt: Optional[str] = Field(
        None,
        alias="initialPrompt",
        description="可选的初始提示词。",
    )
    vad_filter: Optional[bool] = Field(
        False,
        alias="vadFilter",
        description="是否开启 VAD 过滤（仅 faster_whisper 支持）。",
    )
    word_timestamps: bool = Field(
        False,
        alias="wordTimestamps",
        description="是否输出单词级时间戳。",
    )
    diarize: bool = Field(
        False,
        description="是否开启说话人分离（需要 whisperx + HF_TOKEN）。",
    )
    min_speakers: Optional[int] = Field(
        None,
        alias="minSpeakers",
        description="diarize 最小时说话人数。",
    )
    max_speakers: Optional[int] = Field(
        None,
        alias="maxSpeakers",
        description="diarize 最大说话人数。",
    )
    output: Literal["txt", "vtt", "srt", "tsv", "json"] = Field(
        "txt",
        description="输出格式。",
    )

    @field_validator("language")
    def validate_language(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in LANGUAGE_CODES:
            raise ValueError("language 必须是既定的语言编码。")
        return value


@dataclass
class AsyncASRJob:
    job_id: str
    callback_url: str
    task: str
    language: Optional[str]
    output_format: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: str = "pending"
    result: Optional[str] = None
    error: Optional[str] = None
    callback_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "jobId": self.job_id,
            "status": self.status,
            "createdAt": self.created_at.isoformat() + "Z",
            "updatedAt": self.updated_at.isoformat() + "Z",
            "callbackUrl": self.callback_url,
            "task": self.task,
            "language": self.language,
            "outputFormat": self.output_format,
            "result": self.result,
            "error": self.error,
            "callbackError": self.callback_error,
        }


async_jobs: Dict[str, AsyncASRJob] = {}
async_jobs_lock = Lock()

async_job_queue: Optional[asyncio.Queue[tuple[str, AsyncASRRequest]]] = None
queue_worker_tasks: List[asyncio.Task] = []


def _update_job_state(
    job: AsyncASRJob,
    *,
    status: Optional[str] = None,
    result: Optional[str] = None,
    error: Optional[str] = None,
    callback_error: Optional[str] = None,
) -> None:
    with async_jobs_lock:
        if status is not None:
            job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
        if callback_error is not None:
            job.callback_error = callback_error
        job.updated_at = datetime.utcnow()


def _download_audio_from_url(file_url: str) -> bytes:
    request = urllib.request.Request(
        file_url,
        headers={"User-Agent": "whisper-asr-webservice"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"下载音频失败：{exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"下载音频失败：{exc}") from exc


def _transcribe_from_bytes(audio_bytes: bytes, payload: AsyncASRRequest) -> str:
    audio_stream = io.BytesIO(audio_bytes)
    audio = load_audio(audio_stream, payload.encode)
    options = {
        "diarize": payload.diarize,
        "min_speakers": payload.min_speakers,
        "max_speakers": payload.max_speakers,
    }
    output_stream = asr_model.transcribe(
        audio,
        payload.task,
        payload.language,
        payload.initial_prompt,
        payload.vad_filter,
        payload.word_timestamps,
        options,
        payload.output,
    )
    return output_stream.read()


def _post_json(url: str, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status >= 400:
                raise RuntimeError(f"回调地址返回状态 {response.status}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"回调请求失败：{exc}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"回调请求失败：{exc}") from exc


async def _notify_callback(job: AsyncASRJob, payload: dict) -> None:
    try:
        await asyncio.to_thread(_post_json, job.callback_url, payload)
    except Exception as exc:
        _update_job_state(job, callback_error=str(exc))


def _build_callback_payload(job: AsyncASRJob, payload: AsyncASRRequest) -> Dict[str, Optional[str]]:
    return {
        "jobId": job.job_id,
        "status": job.status,
        "task": payload.task,
        "language": payload.language,
        "outputFormat": payload.output,
        "error": job.error,
        "result": job.result,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


async def _process_async_asr_job(job_id: str, payload: AsyncASRRequest) -> None:
    with async_jobs_lock:
        job = async_jobs.get(job_id)
    if job is None:
        return
    _update_job_state(job, status="running")
    try:
        audio_bytes = await asyncio.to_thread(_download_audio_from_url, str(payload.file_url))
        result_text = await asyncio.to_thread(_transcribe_from_bytes, audio_bytes, payload)
        _update_job_state(job, status="completed", result=result_text)
    except Exception as exc:
        _update_job_state(job, status="failed", error=str(exc))
    finally:
        await _notify_callback(job, _build_callback_payload(job, payload))


async def _handle_job_timeout(job_id: str, payload: AsyncASRRequest) -> None:
    with async_jobs_lock:
        job = async_jobs.get(job_id)
    if job is None:
        return
    reason = f"任务超时（限制 {ASYNC_ASR_JOB_TIMEOUT} 秒）"
    _update_job_state(job, status="failed", error=reason)
    await _notify_callback(job, _build_callback_payload(job, payload))


async def _async_job_queue_worker(worker_id: int) -> None:
    assert async_job_queue is not None
    while True:
        job_id, payload = await async_job_queue.get()
        try:
            if ASYNC_ASR_JOB_TIMEOUT > 0:
                await asyncio.wait_for(_process_async_asr_job(job_id, payload), timeout=ASYNC_ASR_JOB_TIMEOUT)
            else:
                await _process_async_asr_job(job_id, payload)
        except asyncio.TimeoutError:
            await _handle_job_timeout(job_id, payload)
        finally:
            async_job_queue.task_done()


@app.on_event("startup")
async def _startup_worker() -> None:
    global async_job_queue, queue_worker_tasks
    async_job_queue = asyncio.Queue()
    queue_worker_tasks = [
        asyncio.create_task(_async_job_queue_worker(worker_id))
        for worker_id in range(ASYNC_ASR_WORKER_COUNT)
    ]


@app.on_event("shutdown")
async def _shutdown_worker() -> None:
    for task in queue_worker_tasks:
        task.cancel()
    for task in queue_worker_tasks:
        with suppress(asyncio.CancelledError):
            await task


projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
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
    result = asr_model.transcribe(
        load_audio(audio_file.file, encode),
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


@app.post("/async-asr", tags=["Endpoints"])
async def async_asr_job(request: AsyncASRRequest):
    job_id = uuid4().hex
    job = AsyncASRJob(
        job_id=job_id,
        callback_url=str(request.callback_url),
        task=request.task,
        language=request.language,
        output_format=request.output,
    )
    with async_jobs_lock:
        async_jobs[job_id] = job
    if async_job_queue is None:
        raise HTTPException(status_code=503, detail="任务队列尚未准备好")
    await async_job_queue.put((job_id, request))
    return {"jobId": job_id, "status": job.status}


@app.get("/queryStatus", tags=["Endpoints"])
async def query_status(job_id: str = Query(..., alias="jobId")):
    with async_jobs_lock:
        job = async_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_dict()


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
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
