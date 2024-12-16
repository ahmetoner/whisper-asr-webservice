import gc
import os
import time
from io import StringIO
from threading import Lock, Thread
from typing import BinaryIO, Union

import torch
import whisper
from whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

model_name = os.getenv("ASR_MODEL", "base")
model_path = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))
model = None
model_lock = Lock()

last_activity_time = time.time()
idle_timeout = int(os.getenv("IDLE_TIMEOUT", 0))  # default to being disabled


def monitor_idleness():
    global model
    if idle_timeout <= 0: return
    while True:
        time.sleep(15)
        if time.time() - last_activity_time > idle_timeout:
            with model_lock:
                release_model()
                break


def load_model():
    global model

    if torch.cuda.is_available():
        model = whisper.load_model(model_name, download_root=model_path).cuda()
    else:
        model = whisper.load_model(model_name, download_root=model_path)

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
    if word_timestamps:
        options_dict["word_timestamps"] = word_timestamps
    with model_lock:
        result = model.transcribe(audio, **options_dict)

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

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(audio, model.dims.n_mels).to(model.device)

    # detect the spoken language
    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)

    return detected_lang_code, probs[max(probs)]


def write_result(result: dict, file: BinaryIO, output: Union[str, None]):
    options = {"max_line_width": 1000, "max_line_count": 10, "highlight_words": False}
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file, options=options)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file, options=options)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file, options=options)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file, options=options)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file, options=options)
    else:
        return "Please select an output method!"
