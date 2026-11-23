import io
import socket
import threading
import time
import wave

import pytest
import uvicorn

from app import webservice
from app.config import CONFIG

openai = pytest.importorskip("openai")


@pytest.fixture(scope="module")
def base_url():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(webservice.app, host="127.0.0.1", port=port, log_config=None))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join()


@pytest.fixture
def client(base_url):
    return openai.OpenAI(base_url=base_url, api_key="secret", max_retries=0)


@pytest.fixture
def audio(pcm_bytes):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm_bytes)
    return ("tone.wav", buffer.getvalue())


def test_sdk_transcription(client, audio):
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio)
    assert transcript.text == "hello world"
    assert transcript.usage.seconds == 1


def test_sdk_verbose_json(client, audio):
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=audio, response_format="verbose_json", timestamp_granularities=["segment"]
    )
    assert transcript.language == "english"
    assert transcript.duration == 1.0
    assert transcript.segments[0].text == " hello world"


def test_sdk_text_and_translation(client, audio):
    assert client.audio.transcriptions.create(model="whisper-1", file=audio, response_format="text") == "hello world\n"
    assert client.audio.translations.create(model="whisper-1", file=audio).text == "hello world"


def test_sdk_streaming(client, audio):
    events = list(client.audio.transcriptions.create(model="gpt-4o-transcribe", file=audio, stream=True))
    assert [event.type for event in events] == ["transcript.text.delta", "transcript.text.done"]
    assert events[-1].text == "hello world"


def test_sdk_models(client):
    assert [model.id for model in client.models.list()][0] == "whisper-1"
    with pytest.raises(openai.NotFoundError):
        client.models.retrieve("gpt-9")


def test_sdk_errors(client, audio, monkeypatch):
    with pytest.raises(openai.BadRequestError, match="Unsupported language"):
        client.audio.transcriptions.create(model="whisper-1", file=audio, language="xx")
    monkeypatch.setattr(CONFIG, "API_KEY", "another-key")
    with pytest.raises(openai.AuthenticationError):
        client.audio.transcriptions.create(model="whisper-1", file=audio)
