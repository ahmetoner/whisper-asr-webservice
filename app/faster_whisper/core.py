
import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch

import whisper
from .utils import model_converter, ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
from faster_whisper import WhisperModel

model_name= os.getenv("ASR_MODEL", "base")
model_path = os.path.join("/root/.cache/faster_whisper", model_name)
model_converter(model_name, model_path)

if torch.cuda.is_available():
    model = WhisperModel(model_path, device="cuda", compute_type="float32")
else:
    model = WhisperModel(model_path, device="cpu", compute_type="int8")
model_lock = Lock()

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    word_timestamps: Union[bool, None],
    output,
):
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if word_timestamps:
        options_dict["word_timestamps"] = True
    with model_lock:   
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
                "text": text
            }
    
    outputFile = StringIO()
    write_result(result, outputFile, output)
    outputFile.seek(0)

    return outputFile

def language_detection(audio):
    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        segments, info = model.transcribe(audio, beam_size=5)
        detected_lang_code = info.language

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
