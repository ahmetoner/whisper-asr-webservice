import logging
from typing import Union

import torch
import whisper

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

logger = logging.getLogger(__name__)


class OpenAIWhisperASR(ASRModel):

    def _load_model(self):
        if CONFIG.CPU_THREADS:
            torch.set_num_threads(CONFIG.CPU_THREADS)
        model = whisper.load_model(name=CONFIG.MODEL_NAME, download_root=CONFIG.MODEL_PATH)
        return model.cuda() if torch.cuda.is_available() else model

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
        if word_timestamps:
            options_dict["word_timestamps"] = word_timestamps
        if CONFIG.BEAM_SIZE:
            options_dict["beam_size"] = CONFIG.BEAM_SIZE
        if options.get("temperature") is not None:
            options_dict["temperature"] = options["temperature"]
        logger.debug("Transcribing with %s", options_dict)
        return self.model.transcribe(audio, **options_dict)

    def _detect_language(self, audio, vad_filter=False):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        detected_lang_code = max(probs, key=probs.get)
        return detected_lang_code, probs[detected_lang_code]
