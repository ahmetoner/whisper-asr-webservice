import contextlib
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict
from typing import BinaryIO, Optional, TextIO

import ffmpeg
import numpy as np
from faster_whisper.utils import format_timestamp

from app.config import CONFIG

logger = logging.getLogger("whisper_asr")


# Raised when an uploaded file cannot be decoded to any audio samples.
# Subclasses RuntimeError so existing callers that caught the previous
# ffmpeg RuntimeError keep working.
class AudioDecodeError(RuntimeError):
    pass


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
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    data = file.read()
    magic = data[:12].hex()
    logger.info("Decoding upload: %d bytes, magic=%s, encode=%s, sr=%d", len(data), magic, encode, sr)
    started = time.monotonic()

    stderr = b""
    if encode:
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        # The upload is spooled to a temp file instead of ffmpeg's stdin:
        # MP4/M4A with the moov index at the end of the file (e.g. written by
        # Android MediaMuxer) needs seekable input. On a pipe, ffmpeg finds
        # the moov but cannot seek back to the media data, and once the file
        # outgrows the IO buffer it silently decodes to 0 samples.
        tmp = tempfile.NamedTemporaryFile(suffix=".upload", delete=False)
        try:
            tmp.write(data)
            tmp.close()
            out, stderr = (
                ffmpeg.input(tmp.name, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise _decode_error("Failed to decode audio", data, magic, e.stderr) from e
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp.name)
    else:
        out = data

    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    if audio.size == 0:
        # Guard here: an empty waveform would otherwise crash deep inside the
        # ASR engine (e.g. whisperx VAD) with a misleading error and a 500.
        raise _decode_error("Audio decoded to 0 samples", data, magic, stderr)
    logger.info(
        "Decoded %d samples (%.2fs at %d Hz) in %d ms",
        audio.size,
        audio.size / sr,
        sr,
        int((time.monotonic() - started) * 1000),
    )
    return audio


def _decode_error(reason: str, data: bytes, magic: str, stderr: bytes) -> AudioDecodeError:
    """Build the client-facing error, log full diagnostics and optionally dump the upload."""
    dump_path = _dump_failed_upload(data, stderr)
    logger.error(
        "%s: upload %d bytes, magic=%s, dump=%s, full ffmpeg output:\n%s",
        reason,
        len(data),
        magic,
        dump_path or "disabled",
        (stderr or b"").decode(errors="replace").strip() or "(none)",
    )
    detail = f"{reason} ({len(data)} bytes, magic={magic})"
    if dump_path:
        detail += f", upload saved to {dump_path}"
    return AudioDecodeError(f"{detail}: {_stderr_excerpt(stderr)}")


def _dump_failed_upload(data: bytes, stderr: bytes) -> Optional[str]:
    """Save the failing upload (and ffmpeg stderr) for offline replay.

    Enabled by setting DEBUG_FAILED_UPLOADS_DIR; off by default so a
    production instance does not archive user audio.
    """
    dump_dir = os.getenv("DEBUG_FAILED_UPLOADS_DIR", "")
    if not dump_dir:
        return None
    try:
        os.makedirs(dump_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S") + "-" + str(time.time_ns() % 1_000_000_000)
        path = os.path.join(dump_dir, f"failed-{stamp}.upload")
        with open(path, "wb") as f:
            f.write(data)
        with open(f"{path[:-len('.upload')]}.stderr.txt", "wb") as f:
            f.write(stderr or b"")
        return path
    except OSError:
        logger.exception("Could not dump failing upload to %s", dump_dir)
        return None


def _stderr_excerpt(stderr: bytes, limit: int = 1500) -> str:
    """Head and tail of ffmpeg stderr: the input analysis lives in the middle,
    the final error usually at the end; keep both when truncating."""
    text = (stderr or b"").decode(errors="replace").strip()
    if not text:
        return "(no ffmpeg diagnostics)"
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + "\n[...]\n" + text[-half:]
