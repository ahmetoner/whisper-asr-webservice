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
    if ASR_ENGINE == "whisperx" and HF_TOKEN == "":
        print("You must set the HF_TOKEN environment variable to download the diarization model used by WhisperX.")

    # Determine the computation device (GPU or CPU)
    DEVICE = os.getenv("ASR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    # Model name to use (e.g., "base", "small", etc.)
    MODEL_NAME = os.getenv("ASR_MODEL", "base")

    # Path to the model directory
    MODEL_PATH = os.getenv("ASR_MODEL_PATH", os.path.join(os.path.expanduser("~"), ".cache", "whisper"))

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

    # Default sample rate for audio input. 16 kHz is commonly used in speech-to-text tasks.
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", 16000))

    # Logging level for the application logger (DEBUG, INFO, WARNING, ERROR).
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # Directory the upload is spooled to before being handed to ffmpeg.
    # Empty means the system temp directory. Point this at real disk if the
    # system temp directory is a RAM-backed tmpfs, as it is in many containers.
    UPLOAD_SPOOL_DIR = os.getenv("UPLOAD_SPOOL_DIR", "") or None

    # When set, uploads that fail to decode are archived in this directory for
    # offline replay. Off by default: it stores raw user audio and grows
    # without bound, so it is a debugging aid, not a production setting.
    DEBUG_FAILED_UPLOADS_DIR = os.getenv("DEBUG_FAILED_UPLOADS_DIR", "")

    # Subtitle output options for whisperx
    SUBTITLE_MAX_LINE_WIDTH = int(os.getenv("SUBTITLE_MAX_LINE_WIDTH", 1000))
    SUBTITLE_MAX_LINE_COUNT = int(os.getenv("SUBTITLE_MAX_LINE_COUNT", 2))
    SUBTITLE_HIGHLIGHT_WORDS = os.getenv("SUBTITLE_HIGHLIGHT_WORDS", "false").lower() == "true"
