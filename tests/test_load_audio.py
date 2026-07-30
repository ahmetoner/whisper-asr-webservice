import io
import logging
import shutil
import subprocess

import pytest

from app.utils import AudioDecodeError, load_audio

ffmpeg_cli = shutil.which("ffmpeg")


@pytest.fixture(scope="module")
def sine_m4a(tmp_path_factory):
    if ffmpeg_cli is None:
        pytest.skip("ffmpeg CLI not available")
    path = tmp_path_factory.mktemp("audio") / "sine.m4a"
    subprocess.run(
        [ffmpeg_cli, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "aac", str(path)],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def long_moov_at_end_m4a(tmp_path_factory):
    # MP4/M4A written by e.g. Android MediaMuxer keeps the moov index at the
    # END of the file. ffmpeg reading a non-seekable pipe can parse the moov
    # but cannot seek back to the media data, so decode yields 0 samples,
    # BUT only once the file outgrows the IO buffer: small files pass, which
    # is why this needs a long (60s, ~500KB) fixture to reproduce.
    if ffmpeg_cli is None:
        pytest.skip("ffmpeg CLI not available")
    path = tmp_path_factory.mktemp("audio") / "long.m4a"
    subprocess.run(
        [ffmpeg_cli, "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=60", "-c:a", "aac", str(path)],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(scope="module")
def video_only_mp4(tmp_path_factory):
    if ffmpeg_cli is None:
        pytest.skip("ffmpeg CLI not available")
    path = tmp_path_factory.mktemp("video") / "video_only.mp4"
    subprocess.run(
        [ffmpeg_cli, "-y", "-f", "lavfi", "-i", "color=black:size=64x64:duration=1", "-an", str(path)],
        check=True,
        capture_output=True,
    )
    return path


def test_decodes_m4a_to_samples(sine_m4a):
    with open(sine_m4a, "rb") as f:
        audio = load_audio(f, encode=True)
    # 2 seconds at 16 kHz, allow generous codec padding slack.
    assert 24000 < audio.size < 40000


def test_decodes_large_moov_at_end_m4a(long_moov_at_end_m4a):
    with open(long_moov_at_end_m4a, "rb") as f:
        audio = load_audio(f, encode=True)
    # 60 seconds at 16 kHz, allow generous codec padding slack.
    assert 900000 < audio.size < 1100000


def test_empty_decode_raises_clear_error():
    # An upload that decodes to zero samples must fail loudly here, not
    # crash later inside an ASR engine with an unrelated message.
    with pytest.raises(AudioDecodeError):
        load_audio(io.BytesIO(b""), encode=False)


def test_no_audio_track_raises_error_with_ffmpeg_stderr(video_only_mp4):
    # A video without any audio track: the error must carry ffmpeg's own
    # diagnostics so the client can see why the file was rejected.
    with open(video_only_mp4, "rb") as f:
        with pytest.raises(AudioDecodeError) as excinfo:
            load_audio(f, encode=True)
    assert "stream" in str(excinfo.value).lower()


def test_error_message_includes_upload_facts():
    # For a debugging build the client-visible error should identify the
    # upload: byte size and magic bytes (real container type, regardless
    # of the filename the client chose).
    data = b"\x00\x01junkjunkjunk"
    with pytest.raises(AudioDecodeError) as excinfo:
        load_audio(io.BytesIO(data), encode=True)
    msg = str(excinfo.value)
    assert f"{len(data)} bytes" in msg
    assert data[:12].hex() in msg


def test_dumps_failing_upload_when_enabled(tmp_path, monkeypatch):
    dump_dir = tmp_path / "dumps"
    monkeypatch.setenv("DEBUG_FAILED_UPLOADS_DIR", str(dump_dir))
    data = b"this is not audio at all"
    with pytest.raises(AudioDecodeError) as excinfo:
        load_audio(io.BytesIO(data), encode=True)
    uploads = list(dump_dir.glob("failed-*.upload"))
    assert len(uploads) == 1
    assert uploads[0].read_bytes() == data
    # ffmpeg stderr is saved next to the upload for offline analysis.
    assert len(list(dump_dir.glob("failed-*.stderr.txt"))) == 1
    # The error tells the operator where the sample was saved.
    assert str(uploads[0]) in str(excinfo.value)


def test_no_dump_when_env_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("DEBUG_FAILED_UPLOADS_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AudioDecodeError):
        load_audio(io.BytesIO(b"junk"), encode=True)
    assert list(tmp_path.iterdir()) == []


def test_success_and_failure_are_logged(sine_m4a, caplog):
    with caplog.at_level(logging.INFO, logger="whisper_asr"):
        with open(sine_m4a, "rb") as f:
            load_audio(f, encode=True)
        with pytest.raises(AudioDecodeError):
            load_audio(io.BytesIO(b"junk"), encode=True)
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert any("samples" in r.getMessage() for r in infos)
    # The full untruncated ffmpeg stderr goes to the server log.
    assert any("ffmpeg" in r.getMessage().lower() for r in errors)
