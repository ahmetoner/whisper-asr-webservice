import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch

import whisper
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON

def initialize_model():
    model_name = os.getenv("ASR_MODEL", "base")
    if torch.cuda.is_available():
        model = whisper.load_model(model_name).cuda()
    else:
        model = whisper.load_model(model_name)
    model_lock = Lock()
    return model, model_lock

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    word_timestamps: Union[bool, None],
    output
):
    model, model_lock = initialize_model()

    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    with model_lock:
        result = model.transcribe(audio, fp16=False , **options_dict)

    outputFile = StringIO()
    write_result(result, outputFile, output)
    outputFile.seek(0)

    return outputFile

def language_detection(audio):
    model, model_lock = initialize_model()

    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)

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
