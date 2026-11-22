import logging
from typing import BinaryIO, Union

import whisper
from faster_whisper import BatchedInferencePipeline, WhisperModel

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASRModel):

    def _load_model(self):
        return WhisperModel(
            model_size_or_path=CONFIG.MODEL_NAME,
            device=CONFIG.DEVICE,
            compute_type=CONFIG.MODEL_QUANTIZATION,
            download_root=CONFIG.MODEL_PATH
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
        beam_size = CONFIG.BEAM_SIZE or 5
        logger.debug("Transcribing with beam_size=%s batch_size=%s %s", beam_size, CONFIG.BATCH_SIZE, options_dict)
        if CONFIG.BATCH_SIZE:
            segment_generator, info = BatchedInferencePipeline(self.model).transcribe(
                audio, beam_size=beam_size, batch_size=CONFIG.BATCH_SIZE, **options_dict
            )
        else:
            segment_generator, info = self.model.transcribe(audio, beam_size=beam_size, **options_dict)
        segments = list(segment_generator)
        text = "".join(segment.text for segment in segments)
        logger.debug("Got %d segments, detected language %s (%.2f)", len(segments), info.language, info.language_probability)
        return {"language": options_dict.get("language", info.language), "segments": segments, "text": text}

    def _detect_language(self, audio):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)
        _, info = self.model.transcribe(audio, beam_size=5)
        return info.language, info.language_probability

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        if output == "srt":
            WriteSRT(ResultWriter).write_result(result, file=file)
        elif output == "vtt":
            WriteVTT(ResultWriter).write_result(result, file=file)
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file)
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file)
        else:
            WriteTXT(ResultWriter).write_result(result, file=file)
