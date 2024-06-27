import os
from io import StringIO
from threading import Lock
from typing import Union, BinaryIO

import torch
import whisper
from faster_whisper import WhisperModel

from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON


model_name = os.getenv("ASR_MODEL", "base")
model_path = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

# More about available quantization levels is here:
#   https://opennmt.net/CTranslate2/quantization.html
if torch.cuda.is_available():
    device = "cuda"
    model_quantization = os.getenv("ASR_QUANTIZATION", "float32")
else:
    device = "cpu"
    model_quantization = os.getenv("ASR_QUANTIZATION", "int8")

model = WhisperModel(
    model_size_or_path=model_name,
    device=device,
    compute_type=model_quantization,
    download_root=model_path
)

model_lock = Lock()

def to_whisper_word(word):
    word_dict = word._asdict()
    word_dict["confidence"] = word_dict.pop("probability")
    return word_dict

def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        output,
):
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
            seg_dict = segment._asdict()
            if "words" in seg_dict:
                seg_dict["words"] = [to_whisper_word(word) for word in seg_dict["words"]]
            segments.append(seg_dict)
            text = text + segment.text
        result = {
            "language": options_dict.get("language", info.language),
            "segments": segments,
            "text": text
        }

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file


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
    options = {
        'max_line_width': 1000,
        'max_line_count': 10,
        'highlight_words': False
    }
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
        return 'Please select an output method!'
