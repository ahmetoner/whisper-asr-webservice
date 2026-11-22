import os

import torch


class CONFIG:
    """
    Configuration class for ASR models.
    Reads environment variables for runtime configuration, with sensible defaults.
    """
    # Determine the ASR engine ('faster_whisper', 'openai_whisper' or 'whisperx')
    ASR_ENGINE = os.getenv("ASR_ENGINE", "openai_whisper")

    # Retrieve Huggingface Token
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    # Determine the computation device (GPU or CPU)
    DEVICE = os.getenv("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Model name to use (e.g., "base", "small", etc.)
    MODEL_NAME = os.getenv("ASR_MODEL", "base")

    # Path to the model directory
    # Defaults to $XDG_CACHE_HOME/whisper (~/.cache/whisper), next to the Hugging Face and torch caches
    CACHE_HOME = os.getenv("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    MODEL_PATH = os.getenv("ASR_MODEL_PATH", os.path.join(CACHE_HOME, "whisper"))

    # WhisperX downloads NLTK sentence tokenizer data on first use; keep it with the other caches instead of ~/nltk_data
    NLTK_DATA = os.environ.setdefault("NLTK_DATA", os.path.join(CACHE_HOME, "nltk_data")).split(os.pathsep)[0]

    # Model quantization level. Defines the precision for model weights:
    #   'float32' - 32-bit floating-point precision (higher precision, slower inference)
    #   'float16' - 16-bit floating-point precision (lower precision, faster inference)
    #   'int8' - 8-bit integer precision (lowest precision, fastest inference)
    # Defaults to 'float32' for GPU availability, 'int8' for CPU.
    MODEL_QUANTIZATION = os.getenv("ASR_QUANTIZATION", "float32" if torch.cuda.is_available() else "int8")
    if MODEL_QUANTIZATION not in {"float32", "float16", "int8"}:
        raise ValueError("Invalid MODEL_QUANTIZATION. Choose 'float32', 'float16', or 'int8'.")

    # Idle timeout in seconds. If set to a non-zero value, the model will be unloaded
    # after being idle for this many seconds. A value of 0 means the model will never be unloaded.
    MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", 0))

    # Optional decoding settings. When unset, each engine keeps its own default.
    BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE")) if os.getenv("ASR_BEAM_SIZE") else None
    BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE")) if os.getenv("ASR_BATCH_SIZE") else None

    # Default value of the vad_filter request parameter (faster_whisper)
    VAD_FILTER = os.getenv("ASR_VAD_FILTER", "false").lower() == "true"

    # Optional API key. When set, requests must send it in the X-API-Key header.
    API_KEY = os.getenv("API_KEY", "")

    # Default sample rate for audio input. 16 kHz is commonly used in speech-to-text tasks.
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

    # Subtitle output options for whisperx
    SUBTITLE_MAX_LINE_WIDTH = int(os.getenv("SUBTITLE_MAX_LINE_WIDTH", 1000))
    SUBTITLE_MAX_LINE_COUNT = int(os.getenv("SUBTITLE_MAX_LINE_COUNT", 2))
    SUBTITLE_HIGHLIGHT_WORDS = os.getenv("SUBTITLE_HIGHLIGHT_WORDS", "false").lower() == "true"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("Invalid LOG_LEVEL. Choose 'DEBUG', 'INFO', 'WARNING' or 'ERROR'.")
