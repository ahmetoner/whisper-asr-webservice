import json
from dataclasses import dataclass

from app.results import build_result, format_result

WHISPERX_SEGMENTS = [
    {
        "start": 0.4,
        "end": 2.0,
        "text": " And so my fellow Americans",
        "speaker": "SPEAKER_00",
        "words": [
            {"word": "And", "start": 0.4, "end": 0.5, "score": 0.6, "speaker": "SPEAKER_00"},
            {"word": "so", "start": 0.6, "end": 0.9, "score": 0.9},
            {"word": "my"},
            {"word": "fellow", "start": 1.2, "end": 1.5, "score": 0.9},
            {"word": "Americans", "start": 1.5, "end": 2.0, "score": 0.9},
        ],
    },
    {"start": 2.5, "end": 3.0, "text": " ask not", "speaker": "SPEAKER_01"},
]


def test_build_result_normalizes_whisperx_segments():
    result = build_result(WHISPERX_SEGMENTS, "en", 3.0, "transcribe")
    first = result["segments"][0]
    assert result["text"] == "And so my fellow Americans ask not"
    assert first["id"] == 0 and result["segments"][1]["id"] == 1
    assert first["tokens"] == [] and first["no_speech_prob"] == 0.0
    assert [word["word"] for word in first["words"]] == [" And", " so", " my", " fellow", " Americans"]
    assert first["words"][0]["probability"] == 0.6
    assert first["words"][2]["start"] == first["words"][2]["end"] == 0.9
    assert first["speaker"] == "SPEAKER_00"


def test_build_result_accepts_faster_whisper_dataclasses():
    @dataclass
    class Word:
        start: float
        end: float
        word: str
        probability: float

    @dataclass
    class Segment:
        id: int
        seek: int
        start: float
        end: float
        text: str
        tokens: list
        avg_logprob: float
        compression_ratio: float
        no_speech_prob: float
        words: list
        temperature: float

    segment = Segment(1, 0, 0.0, 1.0, " Hello.", [50364], -0.2, 1.1, 0.01, [Word(0.0, 1.0, " Hello.", 0.9)], 0.0)
    result = build_result([segment], "en", 1.0, "transcribe")
    assert result["segments"][0]["avg_logprob"] == -0.2
    assert result["segments"][0]["words"] == [{"word": " Hello.", "start": 0.0, "end": 1.0, "probability": 0.9}]


def test_languages_without_spaces_are_joined_directly():
    segments = [{"start": 0.0, "end": 1.0, "text": "你好", "words": [{"word": "你", "start": 0.0, "end": 0.5}]}]
    result = build_result(segments + [{"start": 1.0, "end": 2.0, "text": "世界"}], "zh", 2.0, "transcribe")
    assert result["text"] == "你好世界"
    assert result["segments"][0]["words"][0]["word"] == "你"


def test_whisperx_subtitles_keep_spaces_between_words():
    srt = format_result(build_result(WHISPERX_SEGMENTS, "en", 3.0, "transcribe"), "srt")
    assert "[SPEAKER_00]: And so my fellow Americans" in srt
    assert "00:00:00,400 --> 00:00:02,000" in srt


def test_txt_and_tsv_outputs():
    result = build_result(WHISPERX_SEGMENTS, "en", 3.0, "transcribe")
    assert format_result(result, "txt") == "[SPEAKER_00]: And so my fellow Americans\n[SPEAKER_01]: ask not\n"
    assert format_result(result, "tsv").splitlines()[1] == "400\t2000\t[SPEAKER_00]: And so my fellow Americans"


def test_json_output_uses_the_openai_verbose_json_layout():
    output = json.loads(format_result(build_result(WHISPERX_SEGMENTS, "en", 3.0, "transcribe"), "json"))
    assert list(output) == ["task", "language", "duration", "text", "segments", "words"]
    assert output["language"] == "english"
    assert "words" not in output["segments"][0]
    assert output["segments"][0]["speaker"] == "SPEAKER_00"
    assert output["words"][0] == {"word": "And", "start": 0.4, "end": 0.5, "probability": 0.6, "speaker": "SPEAKER_00"}
