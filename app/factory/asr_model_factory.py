from app.asr_models.asr_model import ASRModel
from app.config import CONFIG


class ASRModelFactory:
    @staticmethod
    def create_asr_model() -> ASRModel:
        # Engines are imported lazily so only the selected engine's dependencies are loaded
        if CONFIG.ASR_ENGINE == "openai_whisper":
            from app.asr_models.openai_whisper_engine import OpenAIWhisperASR

            return OpenAIWhisperASR()
        elif CONFIG.ASR_ENGINE == "faster_whisper":
            from app.asr_models.faster_whisper_engine import FasterWhisperASR

            return FasterWhisperASR()
        elif CONFIG.ASR_ENGINE == "whisperx":
            from app.asr_models.mbain_whisperx_engine import WhisperXASR

            return WhisperXASR()
        elif CONFIG.ASR_ENGINE == "voxtral":
            from app.asr_models.voxtral_engine import VoxtralASR

            return VoxtralASR()
        else:
            raise ValueError(f"Unsupported ASR engine: {CONFIG.ASR_ENGINE}")
