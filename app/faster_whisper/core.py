import logging
import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch

import whisper
from .utils import model_converter, ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
from faster_whisper import WhisperModel

MAPPING = {
    "float16": "float32",
    "int16": "float16",
    "int8_float16": "float16",
    "int8": "float16",
    "int4": "int8",
}

model_quantization = os.getenv("ASR_QUANTIZATION", "float16")
model_name = os.getenv("ASR_MODEL", "base")
base_path = os.path.expanduser('~') + "/.cache/faster_whisper"

model_path = os.path.join(base_path, model_name)
if model_quantization != "float16":
    model_path = os.path.join(base_path, model_name + "_" + model_quantization)

model_converter(model_name, model_path, model_quantization)

if torch.cuda.is_available():
    model = WhisperModel(model_path, device="cuda", compute_type=MAPPING[model_quantization])
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
