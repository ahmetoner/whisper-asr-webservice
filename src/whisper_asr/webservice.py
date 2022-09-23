import uvicorn
from fastapi import FastAPI, Request
import whisper

app = FastAPI()
model = whisper.load_model("base")

@app.post("/asr")
def asr_result(req: Request):
    
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.load_audio("audio.mp3")
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language: {max(probs, key=probs.get)}")

    # decode the audio
    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    return "test"


def start():
    uvicorn.run(app, host="0.0.0.0", port="9000", log_level="info")
