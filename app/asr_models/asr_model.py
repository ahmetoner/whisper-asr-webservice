import gc
import logging
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from io import StringIO
from threading import Lock, Thread
from typing import BinaryIO, Union

import torch

from app.config import CONFIG

logger = logging.getLogger(__name__)


class ASRModel(ABC):
    """Base class for the ASR engines; it owns loading, locking, idle tracking and unloading of the model."""

    def __init__(self):
        self.model = None
        self.model_lock = Lock()
        self.last_activity_time = time.time()

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
        """Transcribes the audio with the loaded model and returns the result."""

    @abstractmethod
    def _detect_language(self, audio) -> tuple[str, float]:
        """Detects the language with the loaded model and returns its code and probability."""

    @abstractmethod
    def write_result(self, result: dict, file: BinaryIO, output: Union[str, None]):
        """Writes the result in the requested output format."""

    def load_model(self):
        logger.info(
            "Loading the %s '%s' model on %s, downloading it first if it is not cached...",
            CONFIG.ASR_ENGINE, CONFIG.MODEL_NAME, CONFIG.DEVICE,
        )
        start = time.time()
        self.model = self._load_model()
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
        with self.loaded_model():
            result = self._transcribe(audio, task, language, initial_prompt, vad_filter, word_timestamps, options)

        output_file = StringIO()
        self.write_result(result, output_file, output)
        output_file.seek(0)
        return output_file

    def language_detection(self, audio):
        with self.loaded_model():
            return self._detect_language(audio)

    def monitor_idleness(self):
        """
        Monitors the idleness of the ASR model and releases the model if it has been idle for too long.
        """
        while True:
            time.sleep(15)
            with self.model_lock:
                if time.time() - self.last_activity_time > CONFIG.MODEL_IDLE_TIMEOUT:
                    self.release_model()
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
        logger.info("Model unloaded after %ss of inactivity", CONFIG.MODEL_IDLE_TIMEOUT)
