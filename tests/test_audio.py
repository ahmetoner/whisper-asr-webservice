import io

import av
import numpy as np
import pytest

from app.utils import load_audio


def encode(pcm_bytes: bytes, container_format: str, codec: str, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format=container_format) as container:
        stream = container.add_stream(codec, rate=rate, layout="mono")
        resampler = av.AudioResampler(
            format=stream.codec_context.format.name, layout="mono", rate=rate, frame_size=stream.codec_context.frame_size
        )
        frame = av.AudioFrame.from_ndarray(np.frombuffer(pcm_bytes, np.int16).reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 16000
        for resampled in [*resampler.resample(frame), *resampler.resample(None)]:
            container.mux(stream.encode(resampled))
        container.mux(stream.encode(None))
    return buffer.getvalue()


# MP4-family files (m4a, mov) are written with their index at the end, which needs a seekable input
@pytest.mark.parametrize(
    "container_format,codec,rate",
    [("ipod", "aac", 16000), ("mov", "aac", 16000), ("flac", "flac", 16000), ("mp3", "libmp3lame", 16000), ("ogg", "libopus", 48000)],
)
def test_load_audio_decodes_containers(pcm_bytes, container_format, codec, rate):
    audio = load_audio(io.BytesIO(encode(pcm_bytes * 20, container_format, codec, rate)))
    assert abs(len(audio) - 20 * 16000) < 16000
    assert np.abs(audio).max() > 0.1


def test_load_audio_raw_pcm(pcm_bytes):
    assert len(load_audio(io.BytesIO(pcm_bytes), encode=False)) == 16000


@pytest.mark.parametrize("data", [b"definitely not audio" * 100, b""])
def test_load_audio_rejects_non_audio(data):
    with pytest.raises(ValueError, match="Failed to load audio"):
        load_audio(io.BytesIO(data))
