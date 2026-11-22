import io
import shutil
import subprocess

import pytest

from app.utils import load_audio

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


@requires_ffmpeg
@pytest.mark.parametrize("extension", ["m4a", "mov", "mp3"])
def test_load_audio_decodes_containers(tmp_path, pcm_bytes, extension):
    source = tmp_path / "tone.pcm"
    source.write_bytes(pcm_bytes * 20)
    target = tmp_path / f"tone.{extension}"
    # MP4-family files written by ffmpeg keep their index at the end, which cannot be read from a pipe
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", source, "-b:a", "256k", target],
        check=True,
    )
    audio = load_audio(io.BytesIO(target.read_bytes()))
    assert abs(len(audio) - 20 * 16000) < 16000


def test_load_audio_raw_pcm(pcm_bytes):
    assert len(load_audio(io.BytesIO(pcm_bytes), encode=False)) == 16000
