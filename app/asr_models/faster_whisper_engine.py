import logging
import time
from io import StringIO
from threading import Thread
from typing import BinaryIO, Union

import whisper
from faster_whisper import BatchedInferencePipeline, WhisperModel

from app.asr_models.asr_model import ASRModel, loading_message
from app.config import CONFIG
from app.utils import ResultWriter, WriteJSON, WriteSRT, WriteTSV, WriteTXT, WriteVTT

logger = logging.getLogger(__name__)


class FasterWhisperASR(ASRModel):

    def load_model(self):

        with loading_message():
            self.model = WhisperModel(
                model_size_or_path=CONFIG.MODEL_NAME,
                device=CONFIG.DEVICE,
                compute_type=CONFIG.MODEL_QUANTIZATION,
                download_root=CONFIG.MODEL_PATH
            )

        Thread(target=self.monitor_idleness, daemon=True).start()

    def transcribe(
            self,
            audio,
            task: Union[str, None],
            language: Union[str, None],
            initial_prompt: Union[str, None],
            vad_filter: Union[bool, None],
            word_timestamps: Union[bool, None],
            options: Union[dict, None],
            output,
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
        with self.model_lock:
            self.last_activity_time = time.time()
            if self.model is None:
                self.load_model()
            segments = []
            text = ""
            beam_size = CONFIG.BEAM_SIZE or 5
            logger.debug("Transcribing with beam_size=%s batch_size=%s %s", beam_size, CONFIG.BATCH_SIZE, options_dict)
            if CONFIG.BATCH_SIZE:
                segment_generator, info = BatchedInferencePipeline(self.model).transcribe(
                    audio, beam_size=beam_size, batch_size=CONFIG.BATCH_SIZE, **options_dict
                )
            else:
                segment_generator, info = self.model.transcribe(audio, beam_size=beam_size, **options_dict)
            for segment in segment_generator:
                segments.append(segment)
                text = text + segment.text
            result = {"language": options_dict.get("language", info.language), "segments": segments, "text": text}
            logger.debug("Got %d segments, detected language %s (%.2f)", len(segments), info.language, info.language_probability)
            self.last_activity_time = time.time()

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):

        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # detect the spoken language
        with self.model_lock:
            self.last_activity_time = time.time()
            if self.model is None:
                self.load_model()
            segments, info = self.model.transcribe(audio, beam_size=5)
            detected_lang_code = info.language
            detected_language_confidence = info.language_probability

        return detected_lang_code, detected_language_confidence

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
