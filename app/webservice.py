from fastapi import FastAPI, File, UploadFile, Query, applications
from fastapi.responses import StreamingResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import whisper
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
from whisper import tokenizer
from faster_whisper import WhisperModel
from .faster_whisper.utils import (
    ResultWriter as faster_whisper_ResultWriter,
    WriteTXT as faster_whisper_WriteTXT,
    WriteSRT as faster_whisper_WriteSRT,
    WriteVTT as faster_whisper_WriteVTT,
    WriteTSV as faster_whisper_WriteTSV,
    WriteJSON as faster_whisper_WriteJSON,
)
import os
from os import path
from pathlib import Path
import ffmpeg
from typing import BinaryIO, Union
import numpy as np
from io import StringIO
from threading import Lock
import torch
import fastapi_offline_swagger_ui
import importlib.metadata 

SAMPLE_RATE=16000
LANGUAGE_CODES=sorted(list(tokenizer.LANGUAGES.keys()))

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

assets_path = fastapi_offline_swagger_ui.__path__[0]
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

whisper_model_name= os.getenv("ASR_MODEL", "base")
faster_whisper_model_path = "/root/.cache/faster_whisper"
if torch.cuda.is_available():
    whisper_model = whisper.load_model(whisper_model_name).cuda()
    faster_whisper_model = WhisperModel(faster_whisper_model_path, device="cuda", compute_type="float16")
else:
    whisper_model = whisper.load_model(whisper_model_name)
    faster_whisper_model = WhisperModel(faster_whisper_model_path)
model_lock = Lock()

def get_modal(faster: bool = False):
    if faster:
        return faster_whisper_model

    return whisper_model


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"

@app.post("/asr", tags=["Endpoints"])
def transcribe(
    audio_file: UploadFile = File(...),
    task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    output : Union[str, None] = Query(default="txt", enum=[ "txt", "vtt", "srt", "tsv", "json"]),
    faster: Union[bool, None] = Query(default=False, enum=[False, True])
):

    result = run_asr(audio_file.file, task, language, initial_prompt, faster)
    filename = audio_file.filename.split('.')[0]
    myFile = StringIO()
    write_result(result, myFile, output, faster)
    myFile.seek(0)
    return StreamingResponse(myFile, media_type="text/plain", 
                            headers={'Content-Disposition': f'attachment; filename="{filename}.{output}"'})


@app.post("/detect-language", tags=["Endpoints"])
def language_detection(
    audio_file: UploadFile = File(...),
    faster: Union[bool, None] = Query(default=False, enum=[False, True]),
):
    # load audio and pad/trim it to fit 30 seconds
    audio = load_audio(audio_file.file)
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        model = get_modal(faster)
        if faster:
            segments, info = model.transcribe(audio, beam_size=5)
            detected_lang_code = info.language
        else:
            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            detected_lang_code = max(probs, key=probs.get)

        result = { "detected_language": tokenizer.LANGUAGES[detected_lang_code],
                  "language_code" : detected_lang_code }

    return result


def run_asr(
    file: BinaryIO, task: Union[str, None], language: Union[str, None],
    initial_prompt: Union[str, None], faster: Union[bool, None],
):
    audio = load_audio(file)
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language    
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt    
    with model_lock:   
        model = get_modal(faster)
        if faster:
            segments = []
            text = ""
            i = 0
            segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict)
            for segment in segment_generator:
                segments.append(segment)
                text = text + segment.text
            result = {
                "language": options_dict.get("language", info.language),
                "segments": segments,
                "text": text,
            }
        else:
            result = model.transcribe(audio, **options_dict)

    return result


def write_result(
    result: dict, file: BinaryIO, output: Union[str, None], faster: Union[bool, None]
):
    if faster:
        if(output == "srt"):
            faster_whisper_WriteSRT(ResultWriter).write_result(result, file = file)
        elif(output == "vtt"):
            faster_whisper_WriteVTT(ResultWriter).write_result(result, file = file)
        elif(output == "tsv"):
            faster_whisper_WriteTSV(ResultWriter).write_result(result, file = file)
        elif(output == "json"):
            faster_whisper_WriteJSON(ResultWriter).write_result(result, file = file)
        elif(output == "txt"):
            faster_whisper_WriteTXT(ResultWriter).write_result(result, file = file)
        else:
            return 'Please select an output method!'
    else:
        if(output == "srt"):
            WriteSRT(ResultWriter).write_result(result, file = file)
        elif(output == "vtt"):
            WriteVTT(ResultWriter).write_result(result, file = file)
        elif(output == "tsv"):
            WriteTSV(ResultWriter).write_result(result, file = file)
        elif(output == "json"):
            WriteJSON(ResultWriter).write_result(result, file = file)
        elif(output == "txt"):
            WriteTXT(ResultWriter).write_result(result, file = file)
        else:
            return 'Please select an output method!'


def load_audio(file: BinaryIO, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
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

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
