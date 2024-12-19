from typing import BinaryIO, Union
from io import StringIO
import whisperx
import whisper
from whisperx.utils import SubtitlesWriter, ResultWriter

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG
from app.utils import WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON


class WhisperXASR(ASRModel):
    def __init__(self):
        self.x_models = dict()

    def load_model(self):

        asr_options = {"without_timestamps": False}
        self.model = whisperx.load_model(
            CONFIG.MODEL_NAME, device=CONFIG.DEVICE, compute_type="float32", asr_options=asr_options
        )

        if CONFIG.HF_TOKEN != "":
            self.diarize_model = whisperx.DiarizationPipeline(use_auth_token=CONFIG.HF_TOKEN, device=CONFIG.DEVICE)

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
        with self.model_lock:
            if self.model is None:
                self.load_model()
            result = self.model.transcribe(audio, **options_dict)

        # Load the required model and cache it
        # If we transcribe models in many different languages, this may lead to OOM propblems
        if result["language"] in self.x_models:
            model_x, metadata = self.x_models[result["language"]]
        else:
            self.x_models[result["language"]] = whisperx.load_align_model(
                language_code=result["language"], device=CONFIG.DEVICE
            )
            model_x, metadata = self.x_models[result["language"]]

        # Align whisper output
        result = whisperx.align(
            result["segments"], model_x, metadata, audio, CONFIG.DEVICE, return_char_alignments=False
        )

        if options.get("diarize", False):
            if CONFIG.HF_TOKEN == "":
                print("Warning! HF_TOKEN is not set. Diarization may not work as expected.")
            min_speakers = options.get("min_speakers", None)
            max_speakers = options.get("max_speakers", None)
            # add min/max number of speakers if known
            diarize_segments = self.diarize_model(audio, min_speakers, max_speakers)
            result = whisperx.assign_word_speakers(diarize_segments, result)

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)

        return output_file

    def language_detection(self, audio):
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.pad_or_trim(audio)

        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

        # detect the spoken language
        with self.model_lock:
            if self.model is None:
                self.load_model()
            _, probs = self.model.detect_language(mel)
        detected_lang_code = max(probs, key=probs.get)

        return detected_lang_code

    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        if output == "srt":
            if CONFIG.HF_TOKEN != "":
                WriteSRT(SubtitlesWriter).write_result(result, file=file, options={})
            else:
                WriteSRT(ResultWriter).write_result(result, file=file, options={})
        elif output == "vtt":
            if CONFIG.HF_TOKEN != "":
                WriteVTT(SubtitlesWriter).write_result(result, file=file, options={})
            else:
                WriteVTT(ResultWriter).write_result(result, file=file, options={})
        elif output == "tsv":
            WriteTSV(ResultWriter).write_result(result, file=file, options={})
        elif output == "json":
            WriteJSON(ResultWriter).write_result(result, file=file, options={})
        elif output == "txt":
            WriteTXT(ResultWriter).write_result(result, file=file, options={})
        else:
            return 'Please select an output method!'
