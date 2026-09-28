import inspect
import sys
from pathlib import Path

import pytest

# conftest.py stubs torch and faster_whisper so the lightweight load_audio
# tests can run without the heavy runtime deps. These tests exercise the
# real installed whisperx package (which needs real torch), so drop any
# stub modules (recognizable by their missing __file__) before importing.
for _name in ("torch", "faster_whisper", "faster_whisper.utils"):
    _mod = sys.modules.get(_name)
    if _mod is not None and getattr(_mod, "__file__", None) is None:
        del sys.modules[_name]

whisperx = pytest.importorskip("whisperx")

from whisperx.diarize import DiarizationPipeline  # noqa: E402

from app.asr_models import mbain_whisperx_engine as engine_mod  # noqa: E402
from app.config import CONFIG  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


class _SpyPipeline:
    """Captures constructor arguments instead of downloading pyannote models."""

    captured_args = None
    captured_kwargs = None

    def __init__(self, *args, **kwargs):
        _SpyPipeline.captured_args = args
        _SpyPipeline.captured_kwargs = kwargs


@pytest.fixture()
def diarize_call(monkeypatch):
    """Run WhisperXASR.load_model() with everything heavyweight mocked out
    and return the (args, kwargs) the engine passed to DiarizationPipeline."""
    _SpyPipeline.captured_args = None
    _SpyPipeline.captured_kwargs = None
    monkeypatch.setattr(engine_mod, "DiarizationPipeline", _SpyPipeline)
    monkeypatch.setattr(engine_mod.whisperx, "load_model", lambda *a, **k: object())
    # load_model spawns the idleness monitor thread; keep the test synchronous.
    monkeypatch.setattr(engine_mod, "Thread", lambda *a, **k: type("T", (), {"start": lambda self: None})())
    monkeypatch.setattr(CONFIG, "HF_TOKEN", "dummy-token")
    asr = engine_mod.WhisperXASR()
    asr.load_model()
    assert _SpyPipeline.captured_kwargs is not None, "DiarizationPipeline was never constructed"
    return _SpyPipeline.captured_args, _SpyPipeline.captured_kwargs


def test_diarization_kwargs_accepted_by_installed_whisperx(diarize_call):
    # The engine must call DiarizationPipeline with arguments the installed
    # whisperx version actually accepts. whisperx 3.8.6 renamed the
    # 'use_auth_token' parameter to 'token', so binding the captured call
    # against the real signature raises TypeError if the app code is stale.
    args, kwargs = diarize_call
    sig = inspect.signature(DiarizationPipeline.__init__)
    sig.bind(None, *args, **kwargs)


def test_diarization_model_pinned_to_pre_upgrade_default(diarize_call):
    # whisperx 3.8.6 changed its default diarization model from
    # pyannote/speaker-diarization-3.1 to pyannote/speaker-diarization-community-1.
    # The service pins the old model explicitly so existing HF tokens (gated
    # per-model) and diarization output stay stable across the upgrade.
    _, kwargs = diarize_call
    assert kwargs.get("model_name") == "pyannote/speaker-diarization-3.1"


def test_gpu_dockerfile_ffmpeg_build_is_redistributable():
    # Not a behavior test: verifying the license of the built ffmpeg would
    # require running the docker build. This guards the configure flags at
    # the source level: --enable-nonfree marks the resulting binaries
    # "nonfree and unredistributable" per ffmpeg's configure, which must
    # never be combined with an image that CI publishes to a registry.
    content = (REPO_ROOT / "Dockerfile.gpu").read_text()
    assert "--enable-nonfree" not in content
