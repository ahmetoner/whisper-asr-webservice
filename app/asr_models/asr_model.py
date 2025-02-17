import gc
import time
from abc import ABC, abstractmethod
from threading import Lock
from typing import Union

import torch

from app.config import CONFIG


class ASRModel(ABC):
    """
    Abstract base class for ASR (Automatic Speech Recognition) models.
    """

    model = None
    model_lock = Lock()
    last_activity_time = time.time()

    def __init__(self):
        pass

    @abstractmethod
    def load_model(self):
        """
        Loads the model from the specified path.
        """
        pass

    @abstractmethod
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
        """
        Perform transcription on the given audio file.
        """
        pass

    @abstractmethod
    def language_detection(self, audio):
        """
        Perform language detection on the given audio file.
        """
        pass

    def monitor_idleness(self):
        """
        Monitors the idleness of the ASR model and releases the model if it has been idle for too long.
        """
        if CONFIG.MODEL_IDLE_TIMEOUT <= 0:
            return
        while True:
            time.sleep(15)
            if time.time() - self.last_activity_time > CONFIG.MODEL_IDLE_TIMEOUT:
                with self.model_lock:
                    self.release_model()
                    break

    def release_model(self):
        """
        Unloads the model from memory and clears any cached GPU memory.
        """
        del self.model
        torch.cuda.empty_cache()
        gc.collect()
        self.model = None
        print("Model unloaded due to timeout")
