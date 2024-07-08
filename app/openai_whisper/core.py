import os
from io import StringIO, BytesIO
from threading import Lock
from typing import BinaryIO, Union, Tuple
import torch
import whisper
from whisper.utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON
import numpy as np
import ffmpeg

model_name = os.getenv("ASR_MODEL", "large-v3")
model_path = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

if torch.cuda.is_available():
    model = whisper.load_model(model_name, download_root=model_path).cuda()
else:
    model = whisper.load_model(model_name, download_root=model_path)
model_lock = Lock()

def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        best_of: Union[int, None],
        beam_size: Union[int, None],
        top_k: Union[int, None],
        output: Union[str, None]
):
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if word_timestamps:
        options_dict["word_timestamps"] = word_timestamps
    if best_of:
        options_dict["best_of"] = best_of
    if beam_size:
        options_dict["beam_size"] = beam_size
    with model_lock:
        result = model.transcribe(audio, **options_dict)

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file
    #     base_result = base_model.transcribe(audio, **options_dict)
    
    # segments = base_result['segments']
    
    # # Improve transcription for each segment with large-v3 model
    # improved_segments = []
    # for segment in segments:
    #     segment_audio = extract_segment_audio(audio, segment['start'], segment['end'])
    #     with model_lock:
    #         segment_result = model.transcribe(segment_audio, **options_dict)
    #         for seg in segment_result['segments']:
    #             improved_segments.append(seg)
    
    # # Combine and sort segments by start time
    # result = {
    #     'text': ' '.join([seg['text'] for seg in improved_segments]),
    #     'segments': sorted(improved_segments, key=lambda x: x['start'])
    # }

    # output_file = StringIO()
    # write_result(result, output_file, output)
    # output_file.seek(0)

    # return output_file

# def extract_segment_audio(audio, start, end, sample_rate=16000):
#     """
#     Extracts a segment of the audio between start and end times.
#     """
#     start_sample = int(start * sample_rate)
#     end_sample = int(end * sample_rate)
#     return audio[start_sample:end_sample]

def language_detection(audio):
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    with model_lock:
        _, probs = model.detect_language(mel)
    detected_lang_code = max(probs, key=probs.get)
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
