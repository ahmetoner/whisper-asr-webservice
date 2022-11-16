import os
import typing
import uvicorn
import time
import json
import urllib
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
from starlette.responses import Response
from .languages import LANGUAGES
import whisper
from whisper.utils import write_srt, write_vtt
import os
import ffmpeg
from typing import BinaryIO, Union
from .languages import LANGUAGES, LANGUAGE_CODES
import numpy as np
from io import StringIO
from threading import Lock
import torch

SAMPLE_RATE=16000

app = FastAPI()

core_model = os.getenv("ASR_MODEL", "medium.en")
if ".en" not in core_model:
    core_model = f"{core_model}.en"

if core_model not in ["tiny.en", "base.en", "small.en", "medium.en"]:
    core_model = "medium.en"

if torch.cuda.is_available():
    model = whisper.load_model(core_model).cuda()
else:
    model = whisper.load_model(core_model)

model_lock = Lock()

def time_ms():
    return time.time_ns() // 1_000_000

class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: typing.Any) -> bytes:
        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


@app.get("/status")
async def status():
    return {"status": "OK"}

@app.get("/transcribe", response_class=PrettyJSONResponse)
def transcribe_file(audio_url: str):
    start_time = time_ms()

    url_filename = audio_url.split("/")[-1]
    tmp_filename = f"{time_ms()}-{url_filename}.mp3"

    urllib.request.urlretrieve(audio_url, tmp_filename)

    audio = whisper.load_audio(tmp_filename)

    options_dict = {
        "language": "en"
    }

    result = model.transcribe(audio, **options_dict)

    if os.path.exists(tmp_filename):
        os.remove(tmp_filename)

    result["model"] = core_model
    result["duration_ms"] = (time_ms() - start_time)

    return result

def start():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="debug")
