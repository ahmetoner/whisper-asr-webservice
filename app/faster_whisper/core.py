import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch

import whisper
from .utils import model_converter, ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
from faster_whisper import WhisperModel

def initialize_model():
    model_name = os.getenv("ASR_MODEL", "base")
    model_path = os.path.join("/root/.cache/faster_whisper", model_name)
    model_converter(model_name, model_path)

    if torch.cuda.is_available():
        model = WhisperModel(model_path, device="cuda", compute_type="float32")
    else:
        model = WhisperModel(model_path, device="cpu", compute_type="int8")
    model_lock = Lock()

    return model, model_lock

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    word_timestamps: Union[bool, None],
    output,
):
    model, model_lock = initialize_model()

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
        segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict, fp16=False)
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
    model, model_lock = initialize_model()

    audio = whisper.pad_or_trim(audio)

    with model_lock:
        segments, info = model.transcribe(audio, beam_size=5, fp16=False)
        detected_lang_code = info.language

    return detected_lang_code

def write_result(
    result: dict, file: BinaryIO, output: Union[str, None]
):
    writers = {
        "srt": WriteSRT(ResultWriter).write_result,
        "vtt": WriteVTT(ResultWriter).write_result,
        "tsv": WriteTSV(ResultWriter).write_result,
        "json": WriteJSON(ResultWriter).write_result,
        "txt": WriteTXT(ResultWriter).write_result
    }

    writer_func = writers.get(output)
    if writer_func:
        writer_func(result, file=file)
    else:
        return 'Please select an output method!'
