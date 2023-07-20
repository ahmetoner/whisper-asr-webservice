import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch

import whisper
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON

model_name= os.getenv("ASR_MODEL", "base")
if torch.cuda.is_available():
    model = whisper.load_model(model_name).cuda()
else:
    model = whisper.load_model(model_name)
model_lock = Lock()

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    word_timestamps: Union[bool, None],
    output
):
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    with model_lock:
        result = model.transcribe(audio, **options_dict)

    outputFile = StringIO()
    write_result(result, outputFile, output)
    outputFile.seek(0)

    return outputFile

def language_detection(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)

    return detected_lang_code

def write_result(
    result: dict, file: BinaryIO, output: Union[str, None]
):
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
