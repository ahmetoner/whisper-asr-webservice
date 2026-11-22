import logging
import os
import subprocess
import sys

import pytest

from app import webservice


def test_index_redirects_to_docs(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"].endswith("/docs")


def test_asr_returns_transcript(client, fake_model, pcm_bytes):
    response = client.post("/asr?encode=false&output=txt", files={"audio_file": ("tone.pcm", pcm_bytes)})
    assert response.status_code == 200
    assert response.text == "hello world\n"
    assert 'filename="tone.pcm.txt"' in response.headers["content-disposition"]
    assert len(fake_model.calls[0]["audio"]) == 16000


@pytest.mark.parametrize(
    "output,content_type",
    [
        ("txt", "text/plain"),
        ("vtt", "text/vtt"),
        ("srt", "application/x-subrip"),
        ("tsv", "text/tab-separated-values"),
        ("json", "application/json"),
    ],
)
def test_asr_content_type_matches_output(client, pcm_bytes, output, content_type):
    response = client.post(f"/asr?encode=false&output={output}", files={"audio_file": ("tone.pcm", pcm_bytes)})
    assert response.headers["content-type"].split(";")[0] == content_type


def test_asr_rejects_unknown_output(client, pcm_bytes):
    response = client.post("/asr?encode=false&output=docx", files={"audio_file": ("tone.pcm", pcm_bytes)})
    assert response.status_code == 422


def test_detect_language(client, pcm_bytes):
    response = client.post("/detect-language?encode=false", files={"audio_file": ("tone.pcm", pcm_bytes)})
    assert response.status_code == 200
    assert response.json() == {"detected_language": "english", "language_code": "en", "confidence": 0.99}


def test_asr_rejects_odd_length_raw_audio(client, fake_model):
    response = client.post("/asr?encode=false", files={"audio_file": ("x.pcm", b"abc")})
    assert response.status_code == 400
    assert fake_model.calls == []


def test_asr_rejects_empty_upload(client, fake_model):
    response = client.post("/asr?encode=false", files={"audio_file": ("x.pcm", b"")})
    assert response.status_code == 400
    assert fake_model.calls == []


def test_openapi_metadata(client):
    info = client.get("/openapi.json").json()["info"]
    assert info["title"] == "Whisper Asr Webservice"
    assert info["contact"]["url"] == "https://github.com/ahmetoner/whisper-asr-webservice/"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_stays_public_when_api_key_configured(client, monkeypatch):
    monkeypatch.setattr(webservice.CONFIG, "API_KEY", "secret")
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("endpoint", ["/asr", "/detect-language"])
def test_api_key_required_when_configured(client, monkeypatch, pcm_bytes, endpoint):
    monkeypatch.setattr(webservice.CONFIG, "API_KEY", "secret")
    files = {"audio_file": ("tone.pcm", pcm_bytes)}
    assert client.post(f"{endpoint}?encode=false", files=files).status_code == 401
    assert client.post(f"{endpoint}?encode=false", files=files, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post(f"{endpoint}?encode=false", files=files, headers={"X-API-Key": "secret"}).status_code == 200


def test_docs_stay_public_when_api_key_configured(client, monkeypatch):
    monkeypatch.setattr(webservice.CONFIG, "API_KEY", "secret")
    assert client.get("/docs").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_model_path_follows_xdg_cache_home(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "ASR_MODEL_PATH"} | {"XDG_CACHE_HOME": str(tmp_path)}
    code = "from app.config import CONFIG; print(CONFIG.MODEL_PATH)"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert result.stdout.strip() == str(tmp_path / "whisper")


def test_requests_are_logged(client, pcm_bytes, caplog):
    caplog.set_level(logging.INFO, logger="app")
    client.post("/asr?encode=false", files={"audio_file": ("tone.pcm", pcm_bytes)})
    client.post("/detect-language?encode=false", files={"audio_file": ("tone.pcm", pcm_bytes)})
    assert "Transcribed 'tone.pcm' (1.0s of audio" in caplog.text
    assert "Detected language en (0.99) in 'tone.pcm'" in caplog.text


def test_invalid_log_level_is_rejected():
    env = os.environ | {"LOG_LEVEL": "LOUD"}
    result = subprocess.run([sys.executable, "-c", "import app.config"], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Invalid LOG_LEVEL" in result.stderr
