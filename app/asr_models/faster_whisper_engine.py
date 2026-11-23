import logging
from typing import Union

from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.vad import VadOptions

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASRModel):

    def _load_model(self):
        return WhisperModel(
            model_size_or_path=CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            download_root=CONFIG.MODEL_PATH,
            cpu_threads=CONFIG.CPU_THREADS or 0,
        )

    def _transcribe(
            self,
            audio,
            task: Union[str, None],
            language: Union[str, None],
            initial_prompt: Union[str, None],
            vad_filter: Union[bool, None],
            word_timestamps: Union[bool, None],
            options: Union[dict, None],
    ):
        options_dict = {"task": task}
        if language:
            options_dict["language"] = language
        if initial_prompt:
            options_dict["initial_prompt"] = initial_prompt
        if vad_filter:
            options_dict["vad_filter"] = True
        if word_timestamps:
            options_dict["word_timestamps"] = True
        if options.get("temperature") is not None:
            options_dict["temperature"] = options["temperature"]
        if CONFIG.PROMPT_RESET_ON_TEMPERATURE is not None:
            options_dict["prompt_reset_on_temperature"] = CONFIG.PROMPT_RESET_ON_TEMPERATURE
        if CONFIG.CONDITION_ON_PREVIOUS_TEXT is not None:
            options_dict["condition_on_previous_text"] = CONFIG.CONDITION_ON_PREVIOUS_TEXT
        vad_parameters = {**self._vad_parameters(), **(options.get("vad") or {})}
        if vad_parameters and (vad_filter or CONFIG.BATCH_SIZE):
            options_dict["vad_parameters"] = vad_parameters
        beam_size = CONFIG.BEAM_SIZE or 5
        logger.debug("Transcribing with beam_size=%s batch_size=%s %s", beam_size, CONFIG.BATCH_SIZE, options_dict)
        if CONFIG.BATCH_SIZE:
            segment_generator, info = BatchedInferencePipeline(self.model).transcribe(
                audio, beam_size=beam_size, batch_size=CONFIG.BATCH_SIZE, **options_dict
            )
        else:
            segment_generator, info = self.model.transcribe(audio, beam_size=beam_size, **options_dict)
        logger.debug("Detected language %s (%.2f)", info.language, info.language_probability)
        return {"language": options_dict.get("language", info.language), "segments": segment_generator}

    def _detect_language(self, audio, vad_filter=False):
        parameters = self._vad_parameters()
        language, probability, _ = self.model.detect_language(
            audio, vad_filter=vad_filter, vad_parameters=VadOptions(**parameters) if parameters else None
        )
        return language, probability

    @staticmethod
    def _vad_parameters():
        settings = {
            "threshold": CONFIG.VAD_THRESHOLD,
            "min_silence_duration_ms": CONFIG.VAD_MIN_SILENCE_MS,
            "speech_pad_ms": CONFIG.VAD_SPEECH_PAD_MS,
        }
        return {key: value for key, value in settings.items() if value is not None}
