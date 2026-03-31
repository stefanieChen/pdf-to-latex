"""Ollama API client for LLM and VLM inference."""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import httpx

logger = logging.getLogger("pdf2latex.ollama")


class OllamaClient:
    """Client for interacting with the Ollama REST API."""

    def __init__(self, base_url: str = "http://localhost:11434", timeout: int = 300):
        """Initialize OllamaClient.

        Args:
            base_url: Ollama server base URL.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)
        self._async_client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    def is_available(self) -> bool:
        """Check if Ollama server is reachable.

        Returns:
            True if server responds, False otherwise.
        """
        try:
            resp = self._client.get("/api/tags")
            return resp.status_code == 200
        except httpx.ConnectError:
            logger.warning("Ollama server not reachable at %s", self.base_url)
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models.

        Returns:
            List of model info dictionaries.
        """
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a specific model is available locally.

        Args:
            model_name: Name of the model.

        Returns:
            True if model is available.
        """
        models = self.list_models()
        return any(m.get("name", "").startswith(model_name) for m in models)

    def generate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        system: Optional[str] = None,
        temperature: float = 0.1,
        stream: bool = False,
    ) -> str:
        """Generate a response from an Ollama model.

        Args:
            model: Model name.
            prompt: User prompt text.
            images: Optional list of image paths or base64 strings for VLM.
            system: Optional system prompt.
            temperature: Sampling temperature.
            stream: Whether to stream the response.

        Returns:
            Generated text response.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        if system:
            payload["system"] = system

        if images:
            payload["images"] = [self._encode_image(img) for img in images]

        logger.debug("Ollama generate: model=%s, prompt_len=%d", model, len(prompt))
        resp = self._client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")

    async def agenerate(
        self,
        model: str,
        prompt: str,
        images: Optional[List[Union[str, Path]]] = None,
        system: Optional[str] = None,
        temperature: float = 0.1,
    ) -> str:
        """Async version of generate.

        Args:
            model: Model name.
            prompt: User prompt text.
            images: Optional list of image paths or base64 strings for VLM.
            system: Optional system prompt.
            temperature: Sampling temperature.

        Returns:
            Generated text response.
        """
        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }

        if system:
            payload["system"] = system

        if images:
            payload["images"] = [self._encode_image(img) for img in images]

        resp = await self._async_client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json().get("response", "")

    def unload_model(self, model: str) -> bool:
        """Unload a model from VRAM.

        Args:
            model: Model name to unload.

        Returns:
            True if successfully unloaded.
        """
        try:
            resp = self._client.post(
                "/api/generate",
                json={"model": model, "keep_alive": 0},
            )
            resp.raise_for_status()
            logger.info("Unloaded Ollama model: %s", model)
            return True
        except Exception as e:
            logger.warning("Failed to unload model %s: %s", model, e)
            return False

    def _encode_image(self, image: Union[str, Path]) -> str:
        """Encode an image to base64 string.

        Args:
            image: Image file path or already base64-encoded string.

        Returns:
            Base64 encoded image string.
        """
        path = Path(image)
        if path.exists():
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        # Assume it's already base64
        return str(image)

    def close(self) -> None:
        """Close HTTP clients."""
        self._client.close()

    async def aclose(self) -> None:
        """Close async HTTP client."""
        await self._async_client.aclose()
