import enum
import uvicorn
from fastapi import FastAPI, File, UploadFile, Query
import whisper
import os
import librosa
from typing import BinaryIO, Union
from .languages import LANGUAGES

app = FastAPI()


model_name= os.getenv("ASR_MODEL", "base")
model = whisper.load_model(model_name)



@app.post("/asr")
def transcribe_file(
                audio_file: UploadFile = File(...),
                task : Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
                language: Union[str, None] = Query(default=None, enum=list(LANGUAGES.keys())),
                ):


    audio, _ = librosa.load(audio_file.file, sr=16000)
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language    
    
    result = model.transcribe(audio, **options_dict)

    return result



def start():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")


    