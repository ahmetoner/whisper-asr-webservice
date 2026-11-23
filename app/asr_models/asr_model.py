import gc
import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock, Thread
from typing import Union

import torch

from app.config import CONFIG
from app.results import build_result, normalize_segment

logger = logging.getLogger(__name__)


class ASRModel(ABC):
    """Base class for the ASR engines; it owns loading, locking, idle tracking, unloading and the result format."""

    def __init__(self):
        self.model = None
        self.model_lock = Lock()
        self.last_activity_time = time.time()
        # CTranslate2 inference slows down once the thread that created the model exits, so this thread never exits
        self._loader = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-loader")

    @abstractmethod
    def _load_model(self):
        """Loads and returns the model."""

    @abstractmethod
    def _transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
    ) -> dict:
        """Transcribes the audio with the loaded model and returns its raw "segments" and the "language" code."""

    @abstractmethod
    def _detect_language(self, audio, vad_filter: bool = False) -> tuple[str, float]:
        """Detects the language with the loaded model and returns its code and probability."""

    def load_model(self):
        logger.info(
            "Loading the %s '%s' model on %s, downloading it first if it is not cached...",
            CONFIG.ASR_ENGINE, CONFIG.MODEL_NAME, CONFIG.DEVICE,
        )
        start = time.time()
        self.model = self._loader.submit(self._load_model).result()
        logger.info("Model loaded in %.1fs", time.time() - start)
        self.last_activity_time = time.time()
        if CONFIG.MODEL_IDLE_TIMEOUT > 0:
            Thread(target=self.monitor_idleness, daemon=True).start()

    @contextmanager
    def loaded_model(self):
        with self.model_lock:
            self.last_activity_time = time.time()
            if self.model is None:
                self.load_model()
            try:
                yield self.model
            finally:
                self.last_activity_time = time.time()
                if CONFIG.MODEL_UNLOAD_AFTER_REQUEST:
                    self.release_model()
                    logger.info("Model unloaded because MODEL_UNLOAD_AFTER_REQUEST is set")

    def transcribe(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
    ) -> dict:
        with self.loaded_model():
            raw = self._transcribe(audio, task, language, initial_prompt, vad_filter, word_timestamps, options)
            segments = list(raw["segments"])
        logger.debug("Got %d segments in language %s", len(segments), raw["language"])
        return build_result(segments, raw["language"], len(audio) / CONFIG.SAMPLE_RATE, task)

    def transcribe_stream(
        self,
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        options: Union[dict, None],
    ):
        """Yields (language, segment) pairs as soon as the engine produces each segment."""
        with self.loaded_model():
            raw = self._transcribe(audio, task, language, initial_prompt, vad_filter, word_timestamps, options)
            for index, segment in enumerate(raw["segments"]):
                yield raw["language"], normalize_segment(segment, index, raw["language"])

    def language_detection(self, audio, vad_filter: bool = False):
        with self.loaded_model():
            return self._detect_language(audio, vad_filter)

    def monitor_idleness(self):
        """
        Monitors the idleness of the ASR model and releases the model if it has been idle for too long.
        """
        while True:
            time.sleep(15)
            with self.model_lock:
                if time.time() - self.last_activity_time > CONFIG.MODEL_IDLE_TIMEOUT:
                    if self.model is not None:
                        self.release_model()
                        logger.info("Model unloaded after %ss of inactivity", CONFIG.MODEL_IDLE_TIMEOUT)
                    break

    def release_model(self):
        """
        Unloads the model from memory and clears any cached GPU memory.
        """
        if self.model is None:
            return
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
