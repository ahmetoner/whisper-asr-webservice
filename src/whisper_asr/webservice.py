import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
import whisper
import os
import ffmpeg
from typing import BinaryIO, Union
from .languages import LANGUAGES
import numpy as np

SAMPLE_RATE=16000

app = FastAPI()


model_name= os.getenv("ASR_MODEL", "base")
model = whisper.load_model(model_name)



@app.post("/asr")
def transcribe_file(
                audio_file: UploadFile = File(...),
                task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
                language: Union[str, None] = Query(default=None, enum=list(LANGUAGES.keys())),
                ):


    audio = load_audio(audio_file.file)
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language    
    
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


    