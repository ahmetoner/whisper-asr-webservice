import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse
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

model_name= os.getenv("ASR_MODEL", "base")

if torch.cuda.is_available():
    model = whisper.load_model(model_name).cuda()
else:
    model = whisper.load_model(model_name)

model_lock = Lock()

@app.post("/asr")
def transcribe_file(
                audio_file: UploadFile = File(...),
                task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
                language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
                ):

    result = run_asr(audio_file.file, task, language)

    return result


@app.post("/detect-language")
def language_detection(
                audio_file: UploadFile = File(...),
                ):

    # load audio and pad/trim it to fit 30 seconds
    audio = load_audio(audio_file.file)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)
    
    result = { "detected_language": LANGUAGES[detected_lang_code],
              "langauge_code" : detected_lang_code }

    return result


@app.post("/get-srt", response_class=StreamingResponse)
def transcribe_file2srt(
                audio_file: UploadFile = File(...),
                task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
                language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
                ):

    result = run_asr(audio_file.file, task, language)
    
    srt_file = StringIO()
    write_srt(result["segments"], file = srt_file)
    srt_file.seek(0)
    srt_filename = f"{audio_file.filename.split('.')[0]}.srt"
    return StreamingResponse(srt_file, media_type="text/plain", 
                             headers={'Content-Disposition': f'attachment; filename="{srt_filename}"'})


@app.post("/get-vtt", response_class=StreamingResponse)
def transcribe_file2vtt(
                audio_file: UploadFile = File(...),
                task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
                language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
                ):

    result = run_asr(audio_file.file, task, language)
    
    vtt_file = StringIO()
    write_vtt(result["segments"], file = vtt_file)
    vtt_file.seek(0)
    vtt_filename = f"{audio_file.filename.split('.')[0]}.vtt"
    return StreamingResponse(vtt_file, media_type="text/plain", 
                             headers={'Content-Disposition': f'attachment; filename="{vtt_filename}"'})


def run_asr(file: BinaryIO, task: Union[str, None], language: Union[str, None] ):
    audio = load_audio(file)
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language    
    with model_lock:   
        result = model.transcribe(audio, **options_dict)
        
    return result


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


def start():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")
