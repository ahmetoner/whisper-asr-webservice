import logging
from typing import BinaryIO, Union

import torch
import whisper
from whisper.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

logger = logging.getLogger(__name__)


class OpenAIWhisperASR(ASRModel):

    def _load_model(self):
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
        logger.debug("Transcribing with %s", options_dict)
        result = self.model.transcribe(audio, **options_dict)
        logger.debug("Got %d segments in language %s", len(result["segments"]), result["language"])
        return result

    def _detect_language(self, audio):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)

        # detect the spoken language
        _, probs = self.model.detect_language(mel)
        detected_lang_code = max(probs, key=probs.get)
        return detected_lang_code, probs[detected_lang_code]

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        options = {"max_line_width": 1000, "max_line_count": 10, "highlight_words": False}
        if output == "srt":
            WriteSRT(ResultWriter).write_result(result, file=file, options=options)
        elif output == "vtt":
            WriteVTT(ResultWriter).write_result(result, file=file, options=options)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options=options)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file, options=options)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file, options=options)
