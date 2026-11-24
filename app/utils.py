import logging
import time
from typing import BinaryIO

import av
import numpy as np

from app.config import CONFIG

logger = logging.getLogger(__name__)


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
