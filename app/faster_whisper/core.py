import gc
import time
from io import StringIO
from threading import Lock, Thread
from typing import BinaryIO, Union

import torch
import whisper
from faster_whisper import WhisperModel

from app.config import CONFIG
from app.faster_whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

model = None
model_lock = Lock()
last_activity_time = time.time()


def monitor_idleness():
    global model
    if CONFIG.MODEL_IDLE_TIMEOUT <= 0: return
    while True:
        time.sleep(15)
        if time.time() - last_activity_time > CONFIG.MODEL_IDLE_TIMEOUT:
            with model_lock:
                release_model()
                break


def load_model():
    global model, device, model_quantization

    model = WhisperModel(
        model_size_or_path=CONFIG.MODEL_NAME,
        device=CONFIG.DEVICE,
        compute_type=CONFIG.MODEL_QUANTIZATION,
        download_root=CONFIG.MODEL_PATH
    )

    Thread(target=monitor_idleness, daemon=True).start()


load_model()


def release_model():
    global model
    del model
    torch.cuda.empty_cache()
    gc.collect()
    model = None
    print("Model unloaded due to timeout")


def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        output,
):
    global last_activity_time
    last_activity_time = time.time()

    with model_lock:
        if model is None: load_model()

    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if vad_filter:
        options_dict["vad_filter"] = True
    if word_timestamps:
        options_dict["word_timestamps"] = True
    with model_lock:
        segments = []
        text = ""
        segment_generator, info = model.transcribe(audio, beam_size=5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {"language": options_dict.get("language", info.language), "segments": segments, "text": text}

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file


def language_detection(audio):
    global last_activity_time
    last_activity_time = time.time()

    with model_lock:
        if model is None: load_model()

    # load audio and pad/trim it to fit 30 seconds
    audio = whisper.pad_or_trim(audio)

    # detect the spoken language
    with model_lock:
        segments, info = model.transcribe(audio, beam_size=5)
        detected_lang_code = info.language
        detected_language_confidence = info.language_probability

    return detected_lang_code, detected_language_confidence


def write_result(result: dict, file: BinaryIO, output: Union[str, None]):
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file)
    else:
        return "Please select an output method!"
