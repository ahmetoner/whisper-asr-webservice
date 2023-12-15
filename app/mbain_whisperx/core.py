import os
from typing import BinaryIO, Union
from io import StringIO
from threading import Lock
import torch
import whisperx
import whisper
from whisperx.utils import SubtitlesWriter, ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON

model_name= os.getenv("ASR_MODEL", "base")
hf_token= os.getenv("HF_TOKEN", "")
x_models = dict()

if torch.cuda.is_available():
    device = "cuda"
    model = whisperx.load_model(model_name, device=device)
    if hf_token != "":
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
else:
    device = "cpu"
    model = whisperx.load_model(model_name, device=device)
    if hf_token != "":
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=hf_token, device=device)
model_lock = Lock()

def transcribe(
    audio,
    task: Union[str, None],
    language: Union[str, None],
    initial_prompt: Union[str, None],
    word_timestamps: Union[bool, None],
    options: Union[dict, None],
    output
):
    options_dict = {"task" : task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    with model_lock:
        result = model.transcribe(audio, **options_dict)

    # Load the required model and cache it
    # If we transcribe models in many differen languages, this may lead to OOM propblems
    if result["language"] in x_models:
        print('Using chached model')
        model_x, metadata = x_models[result["language"]]
    else:
        print(f'Loading model {result["language"]}')
        x_models[result["language"]] = whisperx.load_align_model(language_code=result["language"], device=device)
        model_x, metadata = x_models[result["language"]]

    # Align whisper output    
    result = whisperx.align(result["segments"], model_x, metadata, audio, device, return_char_alignments=False)

    if options["diarize"]:
        if hf_token == "":
            print("Warning! HF_TOKEN is not set. Diarization may not wor as expected.")
        min_speakers = options["min_speakers"]
        max_speakers = options["max_speakers"]
        # add min/max number of speakers if known
        diarize_segments = diarize_model(audio, min_speakers, max_speakers)
        result = whisperx.assign_word_speakers(diarize_segments, result)

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
        if hf_token != "":
            WriteSRT(SubtitlesWriter).write_result(result, file = file, options = {})
        else:
            WriteSRT(ResultWriter).write_result(result, file = file, options = {})
    elif(output == "vtt"):
        if hf_token != "":
            WriteVTT(SubtitlesWriter).write_result(result, file = file, options = {})
        else:
            WriteVTT(ResultWriter).write_result(result, file = file, options = {})
    elif(output == "tsv"):
        WriteTSV(ResultWriter).write_result(result, file = file, options = {})
    elif(output == "json"):
        WriteJSON(ResultWriter).write_result(result, file = file, options = {})
    elif(output == "txt"):
        WriteTXT(ResultWriter).write_result(result, file = file, options = {})
    else:
        return 'Please select an output method!'
