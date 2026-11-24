import logging
from typing import Union

import torch
from transformers import AutoProcessor, VoxtralForConditionalGeneration

from app.asr_models.asr_model import ASRModel
from app.config import CONFIG

logger = logging.getLogger(__name__)

# Voxtral is distributed on the Hugging Face Hub, so a repository id is expected. ASR_MODEL defaults to a Whisper
# size such as "base", which is meaningless here, so fall back to Voxtral-Mini unless a repository id is given.
DEFAULT_MODEL = "mistralai/Voxtral-Mini-3B-2507"

# Voxtral transcribes in a single language and has no built-in language detection, so a language is always required.
DEFAULT_LANGUAGE = "en"


def _resolve_model_id() -> str:
    return CONFIG.MODEL_NAME if "/" in CONFIG.MODEL_NAME else DEFAULT_MODEL


def _dtype():
    if CONFIG.MODEL_QUANTIZATION == "float16":
        return torch.float16
    return torch.bfloat16 if CONFIG.DEVICE == "cuda" else torch.float32


class VoxtralASR(ASRModel):
    """Mistral's Voxtral audio model, used through its transcription request API in Hugging Face transformers."""

    def _load_model(self):
        if CONFIG.CPU_THREADS:
            torch.set_num_threads(CONFIG.CPU_THREADS)
        model_id = _resolve_model_id()
        token = CONFIG.HF_TOKEN or None
        processor = AutoProcessor.from_pretrained(model_id, cache_dir=CONFIG.MODEL_PATH, token=token)
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id, dtype=_dtype(), cache_dir=CONFIG.MODEL_PATH, token=token
        ).to(CONFIG.DEVICE)
        return {"model": model, "processor": processor}

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
        # Voxtral's transcription API only transcribes in the spoken language; it cannot translate or produce
        # word timestamps, and it ignores the prompt and VAD options that the Whisper engines accept.
        if task == "translate":
            logger.warning("Voxtral does not support translation; transcribing in the source language instead")
        language = language or DEFAULT_LANGUAGE

        model, processor = self.model["model"], self.model["processor"]
        inputs = processor.apply_transcription_request(
            language=language,
            audio=audio,
            model_id=_resolve_model_id(),
            sampling_rate=CONFIG.SAMPLE_RATE,
            format=["wav"],
            return_tensors="pt",
        ).to(CONFIG.DEVICE, dtype=model.dtype)

        generate_kwargs = {"max_new_tokens": int(len(audio) / CONFIG.SAMPLE_RATE * 25) + 256}
        temperature = options.get("temperature") if options else None
        if temperature:
            generate_kwargs.update(do_sample=True, temperature=temperature)
        logger.debug("Transcribing with language=%s %s", language, generate_kwargs)

        outputs = model.generate(**inputs, **generate_kwargs)
        text = processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)[0]

        duration = len(audio) / CONFIG.SAMPLE_RATE
        return {"language": language, "segments": [{"start": 0.0, "end": duration, "text": text}]}

    def _detect_language(self, audio, vad_filter=False):
        # Voxtral has no language detection; it is told which language to transcribe. Report the default instead.
        logger.warning("Voxtral has no language detection; reporting %r", DEFAULT_LANGUAGE)
        return DEFAULT_LANGUAGE, 1.0
