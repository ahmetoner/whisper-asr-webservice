import numpy as np
import torch

from app.asr_models import voxtral_engine as vx
from app.config import CONFIG

AUDIO = np.zeros(16000, dtype=np.float32)


class FakeInputs(dict):
    """Stands in for the processor's BatchFeature: unpackable with **, with attribute access and a no-op .to()."""

    def __init__(self, prompt_len):
        super().__init__(input_ids=torch.zeros((1, prompt_len), dtype=torch.long), input_features=torch.zeros((1, 2)))

    def __getattr__(self, name):
        return self[name]

    def to(self, *args, **kwargs):
        return self


class FakeProcessor:
    def __init__(self, text=" hello world", prompt_len=3):
        self.text = text
        self.prompt_len = prompt_len
        self.requests = []

    def apply_transcription_request(self, **kwargs):
        self.requests.append(kwargs)
        return FakeInputs(self.prompt_len)

    def batch_decode(self, tokens, skip_special_tokens=True):
        return [self.text]


class FakeModel:
    dtype = torch.float32

    def __init__(self, processor):
        self.processor = processor
        self.generate_kwargs = None

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        prompt_len = kwargs["input_ids"].shape[1]
        return torch.zeros((1, prompt_len + 5), dtype=torch.long)


def make_engine(processor=None):
    processor = processor or FakeProcessor()
    engine = vx.VoxtralASR()
    engine.model = {"model": FakeModel(processor), "processor": processor}
    return engine


def test_resolve_model_id_falls_back_to_voxtral_mini(monkeypatch):
    monkeypatch.setattr(CONFIG, "MODEL_NAME", "base")
    assert vx._resolve_model_id() == vx.DEFAULT_MODEL
    monkeypatch.setattr(CONFIG, "MODEL_NAME", "mistralai/Voxtral-Small-24B-2507")
    assert vx._resolve_model_id() == "mistralai/Voxtral-Small-24B-2507"


def test_dtype_follows_device_and_quantization(monkeypatch):
    monkeypatch.setattr(CONFIG, "MODEL_QUANTIZATION", "float16")
    assert vx._dtype() == torch.float16
    monkeypatch.setattr(CONFIG, "MODEL_QUANTIZATION", "float32")
    monkeypatch.setattr(CONFIG, "DEVICE", "cpu")
    assert vx._dtype() == torch.float32
    monkeypatch.setattr(CONFIG, "DEVICE", "cuda")
    assert vx._dtype() == torch.bfloat16


def test_load_model_builds_processor_and_model(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        vx.AutoProcessor, "from_pretrained", classmethod(lambda cls, model_id, **kw: ("processor", model_id))
    )

    class Model:
        @classmethod
        def from_pretrained(cls, model_id, **kw):
            seen.update(model_id=model_id, kw=kw)
            return cls()

        def to(self, device):
            seen["device"] = device
            return self

    monkeypatch.setattr(vx, "VoxtralForConditionalGeneration", Model)
    monkeypatch.setattr(CONFIG, "MODEL_NAME", "base")
    loaded = vx.VoxtralASR()._load_model()
    assert loaded["processor"] == ("processor", vx.DEFAULT_MODEL)
    assert seen["model_id"] == vx.DEFAULT_MODEL
    assert seen["device"] == CONFIG.DEVICE


def test_transcribe_returns_one_segment_for_the_whole_audio():
    engine = make_engine()
    raw = engine._transcribe(AUDIO, "transcribe", "en", None, None, None, {})
    assert raw["language"] == "en"
    assert len(raw["segments"]) == 1
    segment = raw["segments"][0]
    assert segment["start"] == 0.0
    assert segment["end"] == 1.0
    assert segment["text"] == " hello world"


def test_transcribe_defaults_the_language_to_english():
    processor = FakeProcessor()
    engine = make_engine(processor)
    raw = engine._transcribe(AUDIO, "transcribe", None, None, None, None, {})
    assert raw["language"] == "en"
    assert processor.requests[0]["language"] == "en"
    assert processor.requests[0]["sampling_rate"] == CONFIG.SAMPLE_RATE
    assert processor.requests[0]["format"] == ["wav"]


def test_transcribe_passes_temperature_to_generate():
    engine = make_engine()
    engine._transcribe(AUDIO, "transcribe", "en", None, None, None, {"temperature": 0.5})
    kwargs = engine.model["model"].generate_kwargs
    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 0.5


def test_translate_warns_and_still_transcribes(caplog):
    engine = make_engine()
    raw = engine._transcribe(AUDIO, "translate", "en", None, None, None, {})
    assert raw["segments"][0]["text"] == " hello world"
    assert "does not support translation" in caplog.text


def test_detect_language_reports_the_default():
    engine = make_engine()
    assert engine._detect_language(AUDIO) == (vx.DEFAULT_LANGUAGE, 1.0)


def test_transcribe_through_the_base_class_builds_the_standard_result():
    engine = make_engine()
    result = engine.transcribe(AUDIO, "transcribe", "en", None, None, None, {})
    assert result["task"] == "transcribe"
    assert result["language"] == "en"
    assert result["duration"] == 1.0
    assert result["text"] == "hello world"
    assert result["segments"][0]["text"] == " hello world"


def test_asr_endpoint_uses_the_voxtral_engine(client, monkeypatch, pcm_bytes):
    from app import service

    monkeypatch.setattr(service, "asr_model", make_engine())
    response = client.post("/asr?encode=false&output=json", files={"audio_file": ("tone.pcm", pcm_bytes)})
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "hello world"
    assert body["language"] == "english"
    assert body["segments"][0]["text"] == " hello world"
