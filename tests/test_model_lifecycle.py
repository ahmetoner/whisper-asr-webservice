import logging

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


def test_whisperx_applies_initial_prompt_for_one_request(monkeypatch):
    from dataclasses import dataclass

    @dataclass
    class Options:
        initial_prompt: str = None

    class Pipeline:
        options = Options()

        def transcribe(self, audio, **kwargs):
            self.seen_prompt = self.options.initial_prompt
            return {"language": "en", "segments": []}

    pipeline = Pipeline()
    monkeypatch.setattr(wx.whisperx, "load_align_model", lambda **kwargs: (None, None))
    monkeypatch.setattr(wx.whisperx, "align", lambda *args, **kwargs: {"segments": []})
    model = wx.WhisperXASR()
    model.model = {"whisperx": pipeline, "diarize_model": None, "align_model": {}}
    model.transcribe(None, "transcribe", None, "Kennedy", None, None, {}, "txt")
    assert pipeline.seen_prompt == "Kennedy"
    assert pipeline.options.initial_prompt is None


def test_whisperx_passes_speaker_limits_by_keyword(monkeypatch):
    class Pipeline:
        options = None

        def transcribe(self, audio, **kwargs):
            return {"language": "en", "segments": []}

    calls = []
    monkeypatch.setattr(wx.whisperx, "load_align_model", lambda **kwargs: (None, None))
    monkeypatch.setattr(wx.whisperx, "align", lambda *args, **kwargs: {"segments": []})
    monkeypatch.setattr(wx.whisperx, "assign_word_speakers", lambda diarization, result: result)
    model = wx.WhisperXASR()
    model.model = {
        "whisperx": Pipeline(),
        "diarize_model": lambda audio, **kwargs: calls.append(kwargs),
        "align_model": {},
    }
    options = {"diarize": True, "min_speakers": 2, "max_speakers": 4}
    model.transcribe(None, "transcribe", None, None, None, None, options, "txt")
    assert calls == [{"min_speakers": 2, "max_speakers": 4}]


def test_load_model_logs_start_and_finish(caplog):
    caplog.set_level(logging.INFO, logger="app")
    FakeASRModel().load_model()
    assert "Loading the" in caplog.text
    assert "Model loaded in" in caplog.text


def test_requests_reload_the_model_after_it_was_released():
    model = FakeASRModel()
    model.load_model()
    model.release_model()
    assert model.transcribe(None, "transcribe", None, None, None, None, {}, "txt").read() == "hello world\n"
    assert model.model == "loaded"
    model.release_model()
    assert model.language_detection(None) == ("en", 0.99)
    assert model.model == "loaded"


def test_engines_do_not_share_state():
    first, second = FakeASRModel(), FakeASRModel()
    first.load_model()
    assert second.model is None
    assert first.model_lock is not second.model_lock
