import uvicorn
from fastapi import FastAPI, File, UploadFile
import whisper
import logging
import os
import shutil
import time
logging.basicConfig(
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("whisper-webservice")

app = FastAPI()

logger.info("Loading the model...")

model = whisper.load_model("base")

logger.info("Model loaded! Webserver ready!")

@app.post("/transcribe")
def transcribe_file(
                audio_file: UploadFile = File(...) 
                ):
    file_location = os.path.join(os.getcwd(),f"files/{audio_file.filename}")
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(audio_file.file, file_object) 
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio(file_location)
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
    uvicorn.run(app, host="0.0.0.0", port="9000", log_level="info")
