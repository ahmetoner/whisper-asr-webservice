import uvicorn
from fastapi import FastAPI, File, UploadFile
import whisper
import os
import librosa
from typing import BinaryIO


app = FastAPI()


model_name= os.getenv("ASR_MODEL", "base")
model = whisper.load_model(model_name)



@app.post("/transcribe")
def transcribe_file(
                audio_file: UploadFile = File(...) 
                ):

    # load audio and pad/trim it to fit 30 seconds
    audio, _ = librosa.load(audio_file.file, sr=16000)
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return { "text": result.text}


def start():
    uvicorn.run(app, host="0.0.0.0", port=9000, log_level="info")

def file2npf32(ifile_obj: BinaryIO):
    """
    Takes a file object, resamples it to 16kHz, mixes it down to mono and returns it as a numpy array of float32
    """

    data, sr = librosa.load(ifile_obj, sr=16000)
    