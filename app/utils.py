import json
import logging
import os
import time
from dataclasses import asdict
from typing import BinaryIO, TextIO

import av
import numpy as np
from faster_whisper.utils import format_timestamp

from app.config import CONFIG

logger = logging.getLogger(__name__)


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


def load_audio(file: BinaryIO, encode=True, sr: int = CONFIG.SAMPLE_RATE):
    """
    Reads an audio or video file object as a mono float32 waveform at the given sample rate.
    With encode=False the input must already be 16-bit little-endian mono PCM at that rate.
    Raises ValueError if the input cannot be decoded as audio.
    """
    start = time.time()
    if encode:
        try:
            out = _decode(file, sr)
        except av.error.FFmpegError as e:
            raise ValueError(f"Failed to load audio: {e.strerror or e}") from e
        logger.debug("Decoded %d samples with PyAV in %.2fs", len(out) // 2, time.time() - start)
    else:
        out = file.read()
        if len(out) % 2:
            raise ValueError("Raw audio must be 16-bit little-endian mono PCM (an even number of bytes)")
        logger.debug("Read %d bytes of raw PCM", len(out))

    if not out:
        raise ValueError("The uploaded file contains no audio")

    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def _decode(file: BinaryIO, sr: int) -> bytes:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=sr)
    out = bytearray()
    with av.open(file, mode="r") as container:
        if not container.streams.audio:
            raise ValueError("The uploaded file contains no audio")
        for packet in container.demux(container.streams.audio[0]):
            try:
                frames = packet.decode()
            except av.error.InvalidDataError:
                continue
            for frame in frames:
                for resampled in resampler.resample(frame):
                    out += resampled.to_ndarray().tobytes()
        for resampled in resampler.resample(None):
            out += resampled.to_ndarray().tobytes()
    return bytes(out)
