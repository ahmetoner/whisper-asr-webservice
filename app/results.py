import dataclasses
import json
from io import StringIO

from whisper.tokenizer import LANGUAGES
from whisper.utils import WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.config import CONFIG

LANGUAGES_WITHOUT_SPACES = {"ja", "zh", "yue", "th", "lo", "my"}
WRITERS = {"txt": WriteTXT, "vtt": WriteVTT, "srt": WriteSRT, "tsv": WriteTSV}


def _as_dict(item) -> dict:
    return dataclasses.asdict(item) if dataclasses.is_dataclass(item) else dict(item)


def _normalize_words(words, language: str, segment_start: float) -> list[dict]:
    normalized = []
    last_end = segment_start
    for word in words or []:
        word = _as_dict(word)
        text = word["word"]
        if language not in LANGUAGES_WITHOUT_SPACES and not text.startswith(" "):
            text = " " + text
        start = word.get("start", last_end)
        end = word.get("end", start)
        item = {"word": text, "start": start, "end": end, "probability": word.get("probability", word.get("score", 0.0))}
        if word.get("speaker"):
            item["speaker"] = word["speaker"]
        normalized.append(item)
        last_end = end
    return normalized


def build_result(segments, language: str, duration: float, task: str) -> dict:
    """Turns the raw segments of any engine into one structure with OpenAI's segment and word fields."""
    normalized = []
    for index, segment in enumerate(segments):
        segment = _as_dict(segment)
        item = {
            "id": segment.get("id", index),
            "seek": segment.get("seek", 0),
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "tokens": segment.get("tokens", []),
            "temperature": segment.get("temperature", 0.0),
            "avg_logprob": segment.get("avg_logprob", 0.0),
            "compression_ratio": segment.get("compression_ratio", 0.0),
            "no_speech_prob": segment.get("no_speech_prob", 0.0),
        }
        words = _normalize_words(segment.get("words"), language, item["start"])
        if words:
            item["words"] = words
        if segment.get("speaker"):
            item["speaker"] = segment["speaker"]
        normalized.append(item)

    separator = "" if language in LANGUAGES_WITHOUT_SPACES else " "
    text = separator.join(segment["text"].strip() for segment in normalized)
    return {"task": task, "language": language, "duration": duration, "text": text, "segments": normalized}


def verbose_json(result: dict) -> dict:
    segments = [{key: value for key, value in segment.items() if key != "words"} for segment in result["segments"]]
    output = {
        "task": result["task"],
        "language": LANGUAGES.get(result["language"], result["language"]),
        "duration": result["duration"],
        "text": result["text"],
        "segments": segments,
    }
    words = [{**word, "word": word["word"].strip()} for segment in result["segments"] for word in segment.get("words", [])]
    if words:
        output["words"] = words
    return output


def _prepare_for_writers(result: dict) -> dict:
    has_words = any(segment.get("words") for segment in result["segments"])
    segments = []
    for segment in result["segments"]:
        if has_words and not segment.get("words"):
            whole = {"word": segment["text"], "start": segment["start"], "end": segment["end"], "probability": 0.0}
            segment = {**segment, "words": [whole]}
        if "speaker" in segment:
            label = f"[{segment['speaker']}]:"
            segment = {**segment, "text": f" {label}{segment['text']}"}
            if segment.get("words"):
                first = segment["words"][0]
                segment["words"] = [{**first, "word": f" {label}{first['word']}"}, *segment["words"][1:]]
        segments.append(segment)
    return {**result, "segments": segments}


def format_result(result: dict, output: str) -> str:
    if output == "json":
        return json.dumps(verbose_json(result), ensure_ascii=False)
    options = {
        "max_line_width": CONFIG.SUBTITLE_MAX_LINE_WIDTH,
        "max_line_count": CONFIG.SUBTITLE_MAX_LINE_COUNT,
        "highlight_words": CONFIG.SUBTITLE_HIGHLIGHT_WORDS,
    }
    file = StringIO()
    WRITERS[output](".").write_result(_prepare_for_writers(result), file=file, options=options)
    return file.getvalue()
