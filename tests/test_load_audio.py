import io
import logging
import shutil
import subprocess

import pytest

from app.config import CONFIG
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


def test_odd_length_raw_pcm_raises_decode_error():
    # int16 samples are 2 bytes, so an odd-length raw payload cannot be
    # parsed. np.frombuffer would raise a bare ValueError (-> 500); this
    # must land on the same 400 path as every other bad upload.
    with pytest.raises(AudioDecodeError):
        load_audio(io.BytesIO(b"\x00\x01\x02"), encode=False)


def test_no_audio_track_raises_error_with_ffmpeg_stderr(video_only_mp4):
    # A video without any audio track: the logged error must carry ffmpeg's
    # own diagnostics so an operator can see why the file was rejected.
    with open(video_only_mp4, "rb") as f:
        with pytest.raises(AudioDecodeError) as excinfo:
            load_audio(f, encode=True)
    assert "stream" in str(excinfo.value).lower()


def test_error_message_includes_upload_facts():
    # Both the logged message and the client-visible detail should identify
    # the upload: byte size and magic bytes (real container type, regardless
    # of the filename the client chose).
    data = b"\x00\x01junkjunkjunk"
    with pytest.raises(AudioDecodeError) as excinfo:
        load_audio(io.BytesIO(data), encode=True)
    for msg in (str(excinfo.value), excinfo.value.client_detail):
        assert f"{len(data)} bytes" in msg
        assert data[:12].hex() in msg


def test_client_detail_hides_server_paths_and_ffmpeg_output(tmp_path, monkeypatch):
    # The HTTP response must not disclose filesystem layout: neither the
    # dump path nor ffmpeg stderr (which names the spool temp file).
    dump_dir = tmp_path / "dumps"
    monkeypatch.setattr(CONFIG, "DEBUG_FAILED_UPLOADS_DIR", str(dump_dir))
    with pytest.raises(AudioDecodeError) as excinfo:
        load_audio(io.BytesIO(b"this is not audio at all"), encode=True)
    detail = excinfo.value.client_detail
    assert str(dump_dir) not in detail
    assert ".upload" not in detail
    assert "ffmpeg" not in detail.lower()


def test_dumps_failing_upload_when_enabled(tmp_path, monkeypatch):
    dump_dir = tmp_path / "dumps"
    monkeypatch.setattr(CONFIG, "DEBUG_FAILED_UPLOADS_DIR", str(dump_dir))
    data = b"this is not audio at all"
    with pytest.raises(AudioDecodeError) as excinfo:
        load_audio(io.BytesIO(data), encode=True)
    uploads = list(dump_dir.glob("failed-*.upload"))
    assert len(uploads) == 1
    assert uploads[0].read_bytes() == data
    # ffmpeg stderr is saved next to the upload for offline analysis.
    assert len(list(dump_dir.glob("failed-*.stderr.txt"))) == 1
    # The logged error tells the operator where the sample was saved.
    assert str(uploads[0]) in str(excinfo.value)


def test_no_dump_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(CONFIG, "DEBUG_FAILED_UPLOADS_DIR", "")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(AudioDecodeError):
        load_audio(io.BytesIO(b"junk"), encode=True)
    assert list(tmp_path.iterdir()) == []


def test_spooled_upload_is_removed(tmp_path, monkeypatch):
    # The temp file handed to ffmpeg must not survive the call, on either
    # the success or the failure path.
    spool = tmp_path / "spool"
    spool.mkdir()
    monkeypatch.setattr(CONFIG, "UPLOAD_SPOOL_DIR", str(spool))
    with pytest.raises(AudioDecodeError):
        load_audio(io.BytesIO(b"junk"), encode=True)
    assert list(spool.iterdir()) == []


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
