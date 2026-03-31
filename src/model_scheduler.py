"""Unified model scheduler for managing VRAM across Ollama and HuggingFace models."""

import logging
import threading
from enum import Enum
from typing import Optional

logger = logging.getLogger("pdf2latex.scheduler")


class ModelType(Enum):
    """Types of models managed by the scheduler."""
    OLLAMA_VLM = "ollama_vlm"
    OLLAMA_LLM = "ollama_llm"
    DETIKZIFY = "detikzify"


class ModelScheduler:
    """Manages VRAM allocation by ensuring only one large model is loaded at a time.

    On 8GB VRAM, DeTikZify (4-bit ~5GB) and Ollama models cannot coexist.
    This scheduler serializes model loading/unloading.
    """

    def __init__(self, ollama_client, detikzify_client, config: dict):
        """Initialize ModelScheduler.

        Args:
            ollama_client: OllamaClient instance.
            detikzify_client: DeTikZifyClient instance.
            config: System configuration dictionary.
        """
        self.ollama = ollama_client
        self.detikzify = detikzify_client
        self.config = config
        self._lock = threading.Lock()
        self._current_model: Optional[ModelType] = None

        self._ollama_vlm_name = config.get("ollama", {}).get("models", {}).get("vlm", "minicpm-v")
        self._ollama_llm_name = config.get("ollama", {}).get("models", {}).get("llm", "qwen2.5:7b")

    @property
    def current_model(self) -> Optional[ModelType]:
        """Currently loaded model type."""
        return self._current_model

    def acquire(self, model_type: ModelType) -> None:
        """Acquire a model by loading it, unloading any conflicting model first.

        Args:
            model_type: The type of model to acquire.
        """
        with self._lock:
            if self._current_model == model_type:
                logger.debug("Model %s already active", model_type.value)
                return

            self._unload_current()
            self._load_model(model_type)
            self._current_model = model_type

    def release(self) -> None:
        """Release the currently loaded model to free VRAM."""
        with self._lock:
            self._unload_current()
            self._current_model = None

    def _load_model(self, model_type: ModelType) -> None:
        """Load a specific model.

        Args:
            model_type: The type of model to load.
        """
        logger.info("Loading model: %s", model_type.value)

        if model_type == ModelType.DETIKZIFY:
            self.detikzify.load()
        elif model_type == ModelType.OLLAMA_VLM:
            # Warmup by sending a minimal request
            self.ollama.generate(self._ollama_vlm_name, "hi", temperature=0.0)
        elif model_type == ModelType.OLLAMA_LLM:
            self.ollama.generate(self._ollama_llm_name, "hi", temperature=0.0)

    def _unload_current(self) -> None:
        """Unload the currently active model."""
        if self._current_model is None:
            return

        logger.info("Unloading model: %s", self._current_model.value)

        if self._current_model == ModelType.DETIKZIFY:
            self.detikzify.unload()
        elif self._current_model == ModelType.OLLAMA_VLM:
            self.ollama.unload_model(self._ollama_vlm_name)
        elif self._current_model == ModelType.OLLAMA_LLM:
            self.ollama.unload_model(self._ollama_llm_name)

    def get_ollama_model_name(self, model_type: ModelType) -> str:
        """Get the Ollama model name for a given model type.

        Args:
            model_type: OLLAMA_VLM or OLLAMA_LLM.

        Returns:
            Model name string.

        Raises:
            ValueError: If model_type is not an Ollama model.
        """
        if model_type == ModelType.OLLAMA_VLM:
            return self._ollama_vlm_name
        elif model_type == ModelType.OLLAMA_LLM:
            return self._ollama_llm_name
        else:
            raise ValueError(f"Not an Ollama model type: {model_type}")
