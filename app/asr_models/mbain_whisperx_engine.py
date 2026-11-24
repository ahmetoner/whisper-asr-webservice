import logging
import os
from dataclasses import replace
from typing import Union

import whisperx
from whisperx.audio import N_SAMPLES
from whisperx.diarize import DiarizationPipeline

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

logger = logging.getLogger(__name__)


class WhisperXASR(ASRModel):

    def _load_model(self):
        # NLTK only downloads into a directory that already exists
        os.makedirs(CONFIG.NLTK_DATA, exist_ok=True)
        asr_options = {"without_timestamps": False}
        if CONFIG.BEAM_SIZE:
            asr_options["beam_size"] = CONFIG.BEAM_SIZE
        if CONFIG.PROMPT_RESET_ON_TEMPERATURE is not None:
            asr_options["prompt_reset_on_temperature"] = CONFIG.PROMPT_RESET_ON_TEMPERATURE
        if CONFIG.CONDITION_ON_PREVIOUS_TEXT is not None:
            asr_options["condition_on_previous_text"] = CONFIG.CONDITION_ON_PREVIOUS_TEXT
        vad_options = None
        if CONFIG.VAD_THRESHOLD is not None:
            vad_options = {"vad_onset": CONFIG.VAD_THRESHOLD, "vad_offset": CONFIG.VAD_THRESHOLD}
        model = {
            'whisperx': whisperx.load_model(
                CONFIG.MODEL_NAME,
                device=CONFIG.DEVICE,
                compute_type=CONFIG.MODEL_QUANTIZATION,
                asr_options=asr_options,
                vad_options=vad_options,
                threads=CONFIG.CPU_THREADS or 4,
            ),
            'diarize_model': None,
            'align_model': {},
        }
        if CONFIG.HF_TOKEN != "":
            model['diarize_model'] = DiarizationPipeline(
                token=CONFIG.HF_TOKEN,
                device=CONFIG.DEVICE
            )
        return model

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

        # whisperx takes the initial prompt and temperature from the pipeline options, not as transcribe() arguments
        pipeline = self.model['whisperx']
        default_options = pipeline.options
        changes = {}
        if initial_prompt:
            changes["initial_prompt"] = initial_prompt
        if options.get("temperature") is not None:
            changes["temperatures"] = [options["temperature"]]
        if changes:
            pipeline.options = replace(default_options, **changes)
        logger.debug("Transcribing with batch_size=%s initial_prompt=%r %s", CONFIG.BATCH_SIZE, initial_prompt, options_dict)
        try:
            result = pipeline.transcribe(audio, batch_size=CONFIG.BATCH_SIZE, **options_dict)
        finally:
            pipeline.options = default_options
        language = result["language"]

        # Load the required model and cache it
        # If we transcribe models in many different languages, this may lead to OOM propblems
        if language not in self.model['align_model']:
            logger.info("Loading the alignment model for language %s", language)
            self.model['align_model'][language] = whisperx.load_align_model(
                language_code=language, device=CONFIG.DEVICE
            )
        model_x, metadata = self.model['align_model'][language]

        # Align whisper output
        result = whisperx.align(
            result["segments"], model_x, metadata, audio, CONFIG.DEVICE, return_char_alignments=False
        )

        if options.get("diarize", False) and self.model['diarize_model'] is not None:
            logger.debug("Diarizing with min_speakers=%s max_speakers=%s", options.get("min_speakers"), options.get("max_speakers"))
            # add min/max number of speakers if known
            diarize_segments = self.model['diarize_model'](
                audio, min_speakers=options.get("min_speakers"), max_speakers=options.get("max_speakers")
            )
            result = whisperx.assign_word_speakers(diarize_segments, result)
        result["language"] = language
        return result

    def _detect_language(self, audio, vad_filter=False):
        if audio.shape[0] < N_SAMPLES:
            logger.warning("Audio is shorter than 30s, language detection may be inaccurate")
        results = self.model['whisperx'].model.detect_language(audio)
        language = results[0]
        language_probability = float(results[1])
        logger.debug("Detected language %s (%s) in the first 30s of audio", language, language_probability)
        return language, language_probability
