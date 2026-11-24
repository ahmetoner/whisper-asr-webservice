import logging
import os
import secrets
from threading import BoundedSemaphore, Event
from typing import Annotated, Optional

from fastapi import HTTPException, Security, UploadFile
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

logger = logging.getLogger(__name__)

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


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer = HTTPBearer(auto_error=False)


def verify_api_key(
    api_key: Annotated[Optional[str], Security(api_key_header)],
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Security(bearer)],
):
    key = api_key or (credentials.credentials if credentials else None)
    if CONFIG.API_KEY and not (key and secrets.compare_digest(key, CONFIG.API_KEY)):
        logger.warning("Rejected a request with a missing or invalid API key")
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


pending_requests = BoundedSemaphore(CONFIG.MAX_PENDING_REQUESTS) if CONFIG.MAX_PENDING_REQUESTS > 0 else None


def limit_pending_requests():
    if pending_requests is None:
        yield
        return
    if not pending_requests.acquire(blocking=False):
        logger.warning("Rejected a request because %s requests are already pending", CONFIG.MAX_PENDING_REQUESTS)
        raise HTTPException(status_code=503, detail="Too many pending requests, try again later")
    try:
        yield
    finally:
        pending_requests.release()


def read_audio(audio_file: UploadFile, encode: bool = True):
    try:
        return load_audio(audio_file.file, encode)
    except ValueError as e:
        logger.warning("Could not read %r: %s", audio_file.filename, e)
        raise HTTPException(status_code=400, detail=str(e)) from e
