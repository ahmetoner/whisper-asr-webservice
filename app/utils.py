import json
import os
import io
import zipfile
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
            # Handle both segment as dict and as object
            text = segment["text"] if isinstance(segment, dict) else segment.text
            print(text.strip(), file=file, flush=True)


class WriteVTT(ResultWriter):
    extension: str = "vtt"

    def write_result(self, result: dict, file: TextIO):
        print("WEBVTT\n", file=file)
        for segment in result["segments"]:
            # Handle both segment as dict and as object
            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
            else:
                start = segment.start
                end = segment.end
                text = segment.text
                
            print(
                f"{format_timestamp(start)} --> {format_timestamp(end)}\n"
                f"{text.strip().replace('-->', '->')}\n",
                file=file,
                flush=True,
            )


class WriteSRT(ResultWriter):
    extension: str = "srt"

    def write_result(self, result: dict, file: TextIO):
        for i, segment in enumerate(result["segments"], start=1):
            # Handle both segment as dict and as object
            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
            else:
                start = segment.start
                end = segment.end
                text = segment.text
                
            # write srt lines
            print(
                f"{i}\n"
                f"{format_timestamp(start, always_include_hours=True, decimal_marker=',')} --> "
                f"{format_timestamp(end, always_include_hours=True, decimal_marker=',')}\n"
                f"{text.strip().replace('-->', '->')}\n",
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
            # Handle both segment as dict and as object
            if isinstance(segment, dict):
                start = segment["start"]
                end = segment["end"]
                text = segment["text"]
            else:
                start = segment.start
                end = segment.end
                text = segment.text
                
            print(round(1000 * start), file=file, end="\t")
            print(round(1000 * end), file=file, end="\t")
            print(text.strip().replace("\t", " "), file=file, flush=True)


class WriteJSON(ResultWriter):
    extension: str = "json"

    def write_result(self, result: dict, file: TextIO):
        if "segments" in result:
            # Check if segments are already dictionaries or need to be converted
            if result["segments"] and not isinstance(result["segments"][0], dict):
                result["segments"] = [asdict(segment) for segment in result["segments"]]
        json.dump(result, file)


class WriteAll:
    """
    Write a transcript to multiple files in all supported formats.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.writers = {
            "txt": WriteTXT(output_dir),
            "vtt": WriteVTT(output_dir),
            "srt": WriteSRT(output_dir),
            "tsv": WriteTSV(output_dir),
            "json": WriteJSON(output_dir)
        }

    def __call__(self, result: dict, audio_path: str):
        for format_name, writer in self.writers.items():
            try:
                writer(result, audio_path)
            except Exception as e:
                print(f"Error in {format_name} writer: {str(e)}")
                # Continue with other formats even if one fails

    def create_zip_bytes(self, result: dict):
        """
        Create a zip file in memory and return its bytes.
        This creates a valid zip file with all transcript formats.
        """
        # Create a new in-memory zip file
        buffer = io.BytesIO()
        
        try:
            # Open the zip file for writing
            with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Write each format to the zip file
                formats = {
                    "txt": WriteTXT,
                    "vtt": WriteVTT,
                    "srt": WriteSRT,
                    "tsv": WriteTSV,
                    "json": WriteJSON
                }
                
                for format_name, writer_class in formats.items():
                    try:
                        # Create a buffer for this format's content
                        output = io.StringIO()
                        
                        # Write the result to the buffer
                        writer = writer_class(self.output_dir)
                        writer.write_result(result, output)
                        
                        # Get the text content and add it to the zip
                        content = output.getvalue().encode('utf-8')  # Convert string to bytes
                        zip_file.writestr(f"transcript.{format_name}", content)
                        
                    except Exception as e:
                        print(f"Error adding {format_name} to zip: {str(e)}")
                        # Continue with other formats
            
            # Reset the buffer position and get the zip bytes
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            print(f"Error creating zip file: {str(e)}")
            # Return an empty buffer if zip creation fails
            return b""


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
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
