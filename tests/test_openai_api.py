import json
from threading import BoundedSemaphore

import pytest

from app import service
from app.config import CONFIG


def transcribe(client, data=None, files=None, headers=None, path="/v1/audio/transcriptions", pcm=b"\0\0" * 16000):
    data = {"model": "whisper-1", **(data or {})}
    files = files or {"file": ("tone.wav", _wav(pcm), "audio/wav")}
    return client.post(path, data=data, files=files, headers=headers or {})


def _wav(pcm: bytes) -> bytes:
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(pcm)
    return buffer.getvalue()


def test_list_and_retrieve_models(client):
    models = client.get("/v1/models").json()
    assert models["object"] == "list"
    assert [model["id"] for model in models["data"]][0] == "whisper-1"
    assert client.get("/v1/models/whisper-1").json()["object"] == "model"


def test_unknown_model_returns_an_openai_error(client):
    response = client.get("/v1/models/gpt-9")
    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "message": "The model 'gpt-9' does not exist",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_json_response(client, pcm_bytes):
    response = transcribe(client, pcm=pcm_bytes)
    assert response.status_code == 200
    assert response.json() == {"text": "hello world", "usage": {"type": "duration", "seconds": 1}}


@pytest.mark.parametrize(
    "response_format,expected",
    [("text", "hello world\n"), ("srt", "1\n00:00:00,000 --> 00:00:01,000\nhello world\n\n"), ("vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\nhello world\n\n")],
)
def test_text_responses(client, response_format, expected):
    response = transcribe(client, {"response_format": response_format})
    assert response.headers["content-type"].startswith("text/plain")
    assert response.text == expected


def test_verbose_json_response(client, fake_model):
    body = transcribe(client, {"response_format": "verbose_json"}).json()
    assert body["task"] == "transcribe" and body["language"] == "english" and body["duration"] == 1.0
    assert body["segments"][0]["text"] == " hello world"
    assert "words" not in body
    assert body["usage"] == {"type": "duration", "seconds": 1}
    assert fake_model.calls[-1]["word_timestamps"] is False


def test_word_timestamp_granularity_enables_word_timestamps(client, fake_model):
    data = {"response_format": "verbose_json", "timestamp_granularities[]": ["word"]}
    body = transcribe(client, data).json()
    assert fake_model.calls[-1]["word_timestamps"] is True
    assert "segments" not in body


def test_request_options_are_passed_to_the_engine(client, fake_model):
    data = {"language": "de", "prompt": "Names:", "keywords[]": ["Kubernetes", "Öner"], "temperature": "0.2"}
    transcribe(client, data)
    call = fake_model.calls[-1]
    assert call["language"] == "de"
    assert call["initial_prompt"] == "Names: Kubernetes, Öner"
    assert call["temperature"] == 0.2
    assert call["task"] == "transcribe"


@pytest.mark.parametrize(
    "languages,expected",
    [(["en"], "en"), (["en-US"], "en"), (["english"], "en"), (["en", "de"], None)],
)
def test_languages_hint(client, fake_model, languages, expected):
    transcribe(client, {"languages[]": languages})
    assert fake_model.calls[-1]["language"] == expected


@pytest.mark.parametrize(
    "data,vad",
    [
        ({"chunking_strategy": "auto"}, {}),
        ({"chunking_strategy": '{"type": "server_vad", "threshold": 0.6, "silence_duration_ms": 900}'},
         {"threshold": 0.6, "min_silence_duration_ms": 900}),
        ({"chunking_strategy[type]": "server_vad", "chunking_strategy[prefix_padding_ms]": "200"}, {"speech_pad_ms": 200}),
    ],
)
def test_chunking_strategy_enables_vad(client, fake_model, data, vad):
    transcribe(client, data)
    assert fake_model.calls[-1]["vad_filter"] is True
    assert fake_model.calls[-1]["vad"] == vad


def test_invalid_chunking_strategy(client):
    response = transcribe(client, {"chunking_strategy": "sometimes"})
    assert response.status_code == 400
    assert response.json()["error"]["param"] == "chunking_strategy"


def test_translation(client, fake_model):
    body = transcribe(client, {"response_format": "json"}, path="/v1/audio/translations").json()
    assert body["text"] == "hello world"
    assert fake_model.calls[-1]["task"] == "translate"


@pytest.mark.parametrize(
    "data,param",
    [
        ({"model": None}, "model"),
        ({"response_format": "docx"}, "response_format"),
        ({"language": "xx"}, "language"),
        ({"temperature": "3"}, "temperature"),
        ({"response_format": "diarized_json"}, "response_format"),
    ],
)
def test_invalid_requests_return_openai_errors(client, data, param):
    data = {key: value for key, value in {"model": "whisper-1", **data}.items() if value is not None}
    response = client.post("/v1/audio/transcriptions", data=data, files={"file": ("a.wav", _wav(b"\0\0" * 16000))})
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["param"] == param


def test_undecodable_file_returns_an_openai_error(client):
    response = transcribe(client, files={"file": ("notes.txt", b"not audio" * 100)})
    assert response.status_code == 400
    assert response.json()["error"]["message"].startswith("Failed to load audio")


def test_streaming(client):
    response = transcribe(client, {"stream": "true"})
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]
    assert events == [
        {"type": "transcript.text.delta", "delta": "hello world"},
        {"type": "transcript.text.done", "text": "hello world", "usage": {"type": "duration", "seconds": 1}},
    ]


def test_api_key_accepts_bearer_and_x_api_key(client, monkeypatch, pcm_bytes):
    monkeypatch.setattr(CONFIG, "API_KEY", "secret")
    assert transcribe(client, headers={"Authorization": "Bearer secret"}).status_code == 200
    assert transcribe(client, headers={"X-API-Key": "secret"}).status_code == 200
    files = {"audio_file": ("tone.pcm", pcm_bytes)}
    assert client.post("/asr?encode=false", files=files, headers={"Authorization": "Bearer secret"}).status_code == 200
    response = transcribe(client, headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "invalid_api_key"


def test_legacy_errors_keep_their_format(client):
    response = client.post("/asr?output=docx", files={"audio_file": ("a.wav", _wav(b"\0\0" * 16000))})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_too_many_pending_requests_returns_an_openai_error(client, monkeypatch):
    semaphore = BoundedSemaphore(1)
    monkeypatch.setattr(service, "pending_requests", semaphore)
    semaphore.acquire()
    response = transcribe(client)
    semaphore.release()
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "server_error"
