import json
import os
import tempfile
from dataclasses import asdict
from typing import BinaryIO, TextIO

import ffmpeg
import numpy as np
from faster_whisper.utils import format_timestamp

from app.config import CONFIG


class ResultWriter:
    extension: str

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def __call__(self, result: dict, audio_path: str):
        audio_basename = os.path.basename(audio_path)
        output_path = os.path.join(self.output_dir, audio_basename + "." + self.extension)

        with open(output_path, "w", encoding="utf-8") as f:
            self.write_result(result, file=f)

    def write_result(self, result: dict, file: TextIO):
        raise NotImplementedError


class WriteTXT(ResultWriter):
    extension: str = "txt"

    def write_result(self, result: dict, file: TextIO):
        for segment in result["segments"]:
            print(segment.text.strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    extension: str = "vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            print(
                f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
                f"{segment.text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteTSV(ResultWriter):
    """
    Write a transcript to a file in TSV (tab-separated values) format containing lines like:
    <start time in integer milliseconds>\t<end time in integer milliseconds>\t<transcript text>

    Using integer milliseconds as start and end times means there's no chance of interference from
    an environment setting a language encoding that causes the decimal in a floating point number
    to appear as a comma; also is faster and more efficient to parse & store, e.g., in C++.
    """

    extension: str = "tsv"

    def write_result(self, result: dict, file: TextIO):
        print("start", "end", "text", sep="\t", file=file)
        for segment in result["segments"]:
            print(round(1000 * segment.start), file=file, end="\t")
            print(round(1000 * segment.end), file=file, end="\t")
            print(segment.text.strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        if "segments" in result:
            result["segments"] = [asdict(segment) for segment in result["segments"]]
        json.dump(result, file)


def load_audio(file, encode: bool = True, sr: int = CONFIG.SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Always writes the input to a temp file so ffmpeg has a seekable source.
    
    Parameters
    ----------
    file : BinaryIO
        The audio file-like object.
    encode : bool
        If true, re-encode audio stream to PCM WAV-like raw s16le before returning.
    sr : int
        The sample rate to resample the audio if necessary.
    
    Returns
    -------
    np.ndarray
        A float32 NumPy array containing the waveform in range [-1.0, 1.0].
    """
    data = file.read()

    if not encode:
        # Raw PCM mode (assume s16le bytes already).
        return np.frombuffer(data, np.int16).astype(np.float32) / 32768.0

    try:
        with tempfile.NamedTemporaryFile(suffix=".audio", delete=True) as tmp:
            tmp.write(data)
            tmp.flush()

            out, _ = (
                ffmpeg
                .input(tmp.name)
                .output("pipe:", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .global_args("-nostdin", "-v", "error")
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
            )
    except ffmpeg.Error as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode(errors='ignore')}") from e

    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
