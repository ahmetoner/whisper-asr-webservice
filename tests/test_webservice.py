import pytest


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
