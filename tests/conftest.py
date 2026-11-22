import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.asr_models.asr_model import ASRModel
from app.factory.asr_model_factory import ASRModelFactory


class FakeASRModel(ASRModel):
    """Stands in for a real engine so the API can be tested without downloading models."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def _load_model(self):
        return "loaded"

    def _transcribe(self, audio, task, language, initial_prompt, vad_filter, word_timestamps, options):
        self.calls.append({"audio": audio, "task": task, "language": language, **options})
        return {"text": "hello world\n"}

    def _detect_language(self, audio):
        return "en", 0.99

    def write_result(self, result, file, output):
        file.write(result["text"])


FAKE_MODEL = FakeASRModel()
ASRModelFactory.create_asr_model = staticmethod(lambda: FAKE_MODEL)

from app import webservice  # noqa: E402  (must be imported after the factory is patched)


@pytest.fixture
def client():
    FAKE_MODEL.calls.clear()
    return TestClient(webservice.app)


@pytest.fixture
def fake_model():
    return FAKE_MODEL


@pytest.fixture
def pcm_bytes():
    """One second of a 440 Hz tone as raw 16 kHz mono s16le PCM."""
    t = np.arange(16000) / 16000
    return (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16).tobytes()
