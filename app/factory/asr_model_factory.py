from app.asr_models.asr_model import ASRModel
from app.asr_models.faster_whisper_engine import FasterWhisperASR
from app.asr_models.mbain_whisperx_engine import WhisperXASR
from app.asr_models.openai_whisper_engine import OpenAIWhisperASR
from app.config import CONFIG


class ASRModelFactory:
    @staticmethod
    def create_asr_model() -> ASRModel:
        if CONFIG.ASR_ENGINE == "openai_whisper":
            return OpenAIWhisperASR()
        elif CONFIG.ASR_ENGINE == "faster_whisper":
            return FasterWhisperASR()
        elif CONFIG.ASR_ENGINE == "whisperx":
            return WhisperXASR()
        else:
            raise ValueError(f"Unsupported ASR engine: {CONFIG.ASR_ENGINE}")
