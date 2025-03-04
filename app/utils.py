import json
import os
import io
from dataclasses import asdict
from typing import BinaryIO, TextIO

import ffmpeg
from pydub import AudioSegment
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


def load_audio(file: BinaryIO, encode=True, sr: int = CONFIG.SAMPLE_RATE, use_ffmpeg: bool = False):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    use_ffmpeg: bool
        If True, use ffmpeg to load audio. If False, use pydub.
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    if encode:
        if use_ffmpeg:
            try:
                # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
                # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
                out, _ = (
                    ffmpeg.input("pipe:", threads=0)
                    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                    .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
                )
                samples = np.frombuffer(out, np.int16).flatten()

            except ffmpeg.Error as e:
                raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        else:
            try:
                # Read audio file with pydub
                audio = AudioSegment.from_file(io.BytesIO(file.read()))
                # Pydub does not support resampling, so we need to convert the frame rate
                if audio.frame_rate != sr:
                    audio = audio.set_frame_rate(sr)
                # Convert audio to mono
                audio = audio.set_channels(1)
                # Convert audio to numpy array
                samples = np.array(audio.get_array_of_samples())

            except Exception as e:
                raise RuntimeError(f"Failed to load audio")
            
    else:
        out = file.read()
        samples = np.frombuffer(out, np.int16).flatten()


    # Convert samples to float32
    samples = samples.astype(np.float32)
    # Normalize the sample values
    samples = samples / 32768.0
    
    return samples


