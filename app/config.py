import os

import torch


def _ctranslate2_has_cuda() -> bool:
    import ctranslate2

    try:
        ctranslate2.get_supported_compute_types("cuda")
        return True
    except ValueError:
        return False


# CTranslate2, used by faster_whisper and whisperx, has no CUDA support in its ARM64 build
CTRANSLATE2_GPU_UNSUPPORTED = torch.cuda.is_available() and not _ctranslate2_has_cuda()


class CONFIG:
    """
    Configuration class for ASR models.
    Reads environment variables for runtime configuration, with sensible defaults.
    """
    # Determine the ASR engine ('faster_whisper', 'openai_whisper', 'whisperx' or 'voxtral')
    ASR_ENGINE = os.getenv("ASR_ENGINE") or ("openai_whisper" if CTRANSLATE2_GPU_UNSUPPORTED else "faster_whisper")

    # Retrieve Huggingface Token
    HF_TOKEN = os.getenv("HF_TOKEN", "")

    # Determine the computation device (GPU or CPU)
    DEVICE = os.getenv("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    if ASR_ENGINE in ("faster_whisper", "whisperx") and DEVICE == "cuda" and CTRANSLATE2_GPU_UNSUPPORTED:
        raise ValueError(
            f"{ASR_ENGINE} cannot use the GPU on this platform because CTranslate2 has no CUDA support for it. "
            "Use ASR_ENGINE=openai_whisper to run on the GPU, or ASR_DEVICE=cpu."
        )

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
    MODEL_UNLOAD_AFTER_REQUEST = os.getenv("MODEL_UNLOAD_AFTER_REQUEST", "false").lower() == "true"

    CPU_THREADS = int(os.getenv("ASR_CPU_THREADS")) if os.getenv("ASR_CPU_THREADS") else None

    # Optional decoding settings. When unset, each engine keeps its own default.
    BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE")) if os.getenv("ASR_BEAM_SIZE") else None
    BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE")) if os.getenv("ASR_BATCH_SIZE") else None

    # Voice activity detection: default of the vad_filter request parameter and the VAD settings
    VAD_THRESHOLD = float(os.getenv("ASR_VAD_THRESHOLD")) if os.getenv("ASR_VAD_THRESHOLD") else None
    VAD_MIN_SILENCE_MS = int(os.getenv("ASR_VAD_MIN_SILENCE_MS")) if os.getenv("ASR_VAD_MIN_SILENCE_MS") else None
    VAD_SPEECH_PAD_MS = int(os.getenv("ASR_VAD_SPEECH_PAD_MS")) if os.getenv("ASR_VAD_SPEECH_PAD_MS") else None
    CONDITION_ON_PREVIOUS_TEXT = (
        os.getenv("ASR_CONDITION_ON_PREVIOUS_TEXT").lower() == "true" if os.getenv("ASR_CONDITION_ON_PREVIOUS_TEXT") else None
    )
    PROMPT_RESET_ON_TEMPERATURE = (
        float(os.getenv("ASR_PROMPT_RESET_ON_TEMPERATURE")) if os.getenv("ASR_PROMPT_RESET_ON_TEMPERATURE") else None
    )
    VAD_FILTER = os.getenv("ASR_VAD_FILTER", "false").lower() == "true"

    # Optional API key. When set, requests must send it in the X-API-Key header.
    API_KEY = os.getenv("API_KEY", "")

    # Default sample rate for audio input. 16 kHz is commonly used in speech-to-text tasks.
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

    # Subtitle output options
    SUBTITLE_MAX_LINE_WIDTH = int(os.getenv("SUBTITLE_MAX_LINE_WIDTH")) if os.getenv("SUBTITLE_MAX_LINE_WIDTH") else None
    SUBTITLE_MAX_LINE_COUNT = int(os.getenv("SUBTITLE_MAX_LINE_COUNT")) if os.getenv("SUBTITLE_MAX_LINE_COUNT") else None
    SUBTITLE_HIGHLIGHT_WORDS = os.getenv("SUBTITLE_HIGHLIGHT_WORDS", "false").lower() == "true"

    MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", 0))
    MAX_PENDING_REQUESTS = int(os.getenv("MAX_PENDING_REQUESTS", 0))

    CORS_ALLOW_ORIGINS = [origin.strip() for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if origin.strip()]
    ROOT_PATH = os.getenv("ROOT_PATH", "").rstrip("/")

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    if LOG_LEVEL not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError("Invalid LOG_LEVEL. Choose 'DEBUG', 'INFO', 'WARNING' or 'ERROR'.")
