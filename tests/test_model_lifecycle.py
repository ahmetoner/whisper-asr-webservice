from app.asr_models import mbain_whisperx_engine as wx
from tests.conftest import FakeASRModel


def test_release_model_is_idempotent():
    model = FakeASRModel()
    model.load_model()
    model.release_model()
    model.release_model()
    assert model.model is None


def test_whisperx_reloads_after_idle_release(monkeypatch):
    monkeypatch.setattr(wx.whisperx, "load_model", lambda *args, **kwargs: "pipeline")
    monkeypatch.setattr(wx.CONFIG, "HF_TOKEN", "")
    model = wx.WhisperXASR()
    model.load_model()
    model.release_model()
    assert model.model is None
    model.load_model()
    assert model.model == {"whisperx": "pipeline", "diarize_model": None, "align_model": {}}
