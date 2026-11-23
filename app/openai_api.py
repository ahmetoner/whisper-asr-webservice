import json
import logging
import math
import time
from typing import Annotated, Literal, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Security, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse
from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

from app import service
from app.config import CONFIG
from app.results import (
    LANGUAGES_WITHOUT_SPACES,
    diarized_json,
    format_result,
    join_text,
    speaker_labels,
    verbose_json,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", dependencies=[Security(service.verify_api_key)])

MODEL_IDS = list(dict.fromkeys(["whisper-1", CONFIG.MODEL_NAME]))
TranscriptionFormat = Literal["json", "text", "srt", "verbose_json", "vtt", "diarized_json"]
TranslationFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]


def openai_error(status_code: int, message: str, param: Optional[str] = None, code: Optional[str] = None):
    return HTTPException(status_code=status_code, detail={"message": message, "param": param, "code": code})


def _language_code(value: str) -> str:
    code = value.strip().lower().replace("_", "-").split("-")[0]
    code = TO_LANGUAGE_CODE.get(code, code)
    if code not in LANGUAGES:
        raise openai_error(400, f"Unsupported language: {value}", param="language", code="unsupported_language")
    return code


def _requested_language(language: Optional[str], languages: Optional[list[str]]) -> Optional[str]:
    if language:
        return _language_code(language)
    codes = {_language_code(item) for item in languages or []}
    # Whisper transcribes in a single language, so several hints fall back to detection
    return codes.pop() if len(codes) == 1 else None


def _prompt(prompt: Optional[str], keywords: Optional[list[str]]) -> Optional[str]:
    parts = [part for part in (prompt, ", ".join(keywords or [])) if part]
    return " ".join(parts) or None


def _usage(duration: float) -> dict:
    return {"type": "duration", "seconds": math.ceil(duration)}


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _model_object(model_id: str) -> dict:
    return {"id": model_id, "object": "model", "created": 0, "owned_by": "whisper-asr-webservice"}


@router.get("/models", tags=["Models"])
def list_models():
    return {"object": "list", "data": [_model_object(model_id) for model_id in MODEL_IDS]}


@router.get("/models/{model_id}", tags=["Models"])
def retrieve_model(model_id: str):
    if model_id not in MODEL_IDS:
        raise openai_error(404, f"The model '{model_id}' does not exist", param="model", code="model_not_found")
    return _model_object(model_id)


def _respond(result: dict, response_format: str, granularities: list[str]):
    usage = _usage(result["duration"])
    if response_format == "json":
        return {"text": result["text"], "usage": usage}
    if response_format == "text":
        return PlainTextResponse(result["text"] + "\n")
    if response_format in ("srt", "vtt"):
        return PlainTextResponse(format_result(result, response_format))
    if response_format == "diarized_json":
        return {**diarized_json(result), "usage": usage}
    body = verbose_json(result)
    if "segment" not in granularities:
        body.pop("segments")
    if "word" not in granularities:
        body.pop("words", None)
    return {**body, "usage": usage}


def _stream(audio, task, language, prompt, vad_filter, word_timestamps, options, diarize: bool, filename: str):
    start = time.time()
    texts, segments, detected = [], [], language
    for detected, segment in service.asr_model.transcribe_stream(
        audio, task, language, prompt, vad_filter, word_timestamps, options
    ):
        delta = segment["text"].strip()
        if texts and detected not in LANGUAGES_WITHOUT_SPACES:
            delta = " " + delta
        texts.append(segment["text"])
        segments.append(segment)
        yield _sse({"type": "transcript.text.delta", "delta": delta})
    if diarize:
        labels = speaker_labels(segments)
        for index, segment in enumerate(segments, start=1):
            yield _sse({
                "type": "transcript.text.segment",
                "id": f"seg_{index:03d}",
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "speaker": labels.get(segment.get("speaker"), "unknown"),
            })
    duration = len(audio) / CONFIG.SAMPLE_RATE
    yield _sse({"type": "transcript.text.done", "text": join_text(texts, detected or ""), "usage": _usage(duration)})
    logger.info("Streamed %r (%.1fs of audio, task %s) in %.1fs", filename, duration, task, time.time() - start)


def _run(file, task, language, prompt, temperature, response_format, granularities, stream, chunking_strategy):
    diarize = response_format == "diarized_json"
    if diarize and not (CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN):
        raise openai_error(
            400, "diarized_json requires ASR_ENGINE=whisperx with HF_TOKEN set", param="response_format"
        )
    start = time.time()
    audio = service.read_audio(file)
    vad_filter = CONFIG.VAD_FILTER or chunking_strategy is not None
    word_timestamps = response_format == "verbose_json" and "word" in granularities
    options = {"diarize": diarize, "min_speakers": None, "max_speakers": None, "temperature": temperature}
    if stream:
        return StreamingResponse(
            _stream(audio, task, language, prompt, vad_filter, word_timestamps, options, diarize, file.filename),
            media_type="text/event-stream",
        )
    result = service.asr_model.transcribe(audio, task, language, prompt, vad_filter, word_timestamps, options)
    logger.info(
        "Transcribed %r (%.1fs of audio, task %s, language %s, format %s) in %.1fs",
        file.filename, result["duration"], task, language or "auto", response_format, time.time() - start,
    )
    return _respond(result, response_format, granularities)


@router.post("/audio/transcriptions", tags=["Audio"], dependencies=[Depends(service.limit_pending_requests)])
def create_transcription(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    language: Annotated[Optional[str], Form()] = None,
    languages: Annotated[Optional[list[str]], Form(alias="languages[]")] = None,
    prompt: Annotated[Optional[str], Form()] = None,
    keywords: Annotated[Optional[list[str]], Form(alias="keywords[]")] = None,
    response_format: Annotated[TranscriptionFormat, Form()] = "json",
    temperature: Annotated[Optional[float], Form(ge=0, le=1)] = None,
    timestamp_granularities: Annotated[
        list[Literal["word", "segment"]], Form(alias="timestamp_granularities[]")
    ] = ["segment"],  # noqa: B006
    stream: Annotated[bool, Form()] = False,
    chunking_strategy: Annotated[Optional[str], Form()] = None,
    include: Annotated[Optional[list[str]], Form(alias="include[]")] = None,
    known_speaker_names: Annotated[Optional[list[str]], Form(alias="known_speaker_names[]")] = None,
    known_speaker_references: Annotated[Optional[list[str]], Form(alias="known_speaker_references[]")] = None,
):
    return _run(
        file,
        "transcribe",
        _requested_language(language, languages),
        _prompt(prompt, keywords),
        temperature,
        response_format,
        timestamp_granularities,
        stream,
        chunking_strategy,
    )


@router.post("/audio/translations", tags=["Audio"], dependencies=[Depends(service.limit_pending_requests)])
def create_translation(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()],
    prompt: Annotated[Optional[str], Form()] = None,
    response_format: Annotated[TranslationFormat, Form()] = "json",
    temperature: Annotated[Optional[float], Form(ge=0, le=1)] = None,
):
    return _run(file, "translate", None, prompt, temperature, response_format, ["segment"], False, None)
