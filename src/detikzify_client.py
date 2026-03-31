"""DeTikZify client for figure-to-TikZ conversion."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger("pdf2latex.detikzify")


class DeTikZifyClient:
    """Client wrapper for DeTikZify model inference.

    Handles model loading/unloading and provides sample/simulate APIs
    for converting images to TikZ code.
    """

    def __init__(
        self,
        model_name: str = "nllg/detikzify-v2.5-8b",
        torch_dtype: str = "bfloat16",
        quantize_4bit: bool = True,
        device_map: str = "auto",
    ):
        """Initialize DeTikZifyClient.

        Args:
            model_name: HuggingFace model name or path.
            torch_dtype: Torch data type for model weights.
            quantize_4bit: Whether to use 4-bit quantization.
            device_map: Device mapping strategy.
        """
        self.model_name = model_name
        self.torch_dtype = torch_dtype
        self.quantize_4bit = quantize_4bit
        self.device_map = device_map
        self._pipeline = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded

    def load(self) -> None:
        """Load the DeTikZify model into memory/VRAM."""
        if self._loaded:
            logger.debug("DeTikZify model already loaded")
            return

        try:
            from detikzify.model import load as detikzify_load
            from detikzify.infer import DetikzifyPipeline

            logger.info("Loading DeTikZify model: %s (4-bit=%s)", self.model_name, self.quantize_4bit)

            load_kwargs: Dict[str, Any] = {
                "model_name_or_path": self.model_name,
                "device_map": self.device_map,
                "torch_dtype": self.torch_dtype,
            }

            if self.quantize_4bit:
                load_kwargs["load_in_4bit"] = True

            self._pipeline = DetikzifyPipeline(*detikzify_load(**load_kwargs))
            self._loaded = True
            logger.info("DeTikZify model loaded successfully")

        except ImportError:
            logger.error(
                "DeTikZify not installed. Run: pip install 'detikzify @ git+https://github.com/potamides/DeTikZify'"
            )
            raise
        except Exception as e:
            logger.error("Failed to load DeTikZify model: %s", e)
            raise

    def unload(self) -> None:
        """Unload model from VRAM to free memory."""
        if not self._loaded:
            return

        import gc
        import torch

        del self._pipeline
        self._pipeline = None
        self._loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("DeTikZify model unloaded")

    def sample(self, image: Any) -> Optional[str]:
        """Generate a single TikZ program from an image.

        Args:
            image: PIL Image, file path, or URL.

        Returns:
            TikZ code string if successful, None otherwise.
        """
        if not self._loaded:
            self.load()

        image = self._prepare_image(image)
        logger.debug("DeTikZify sample: generating TikZ code")

        try:
            fig = self._pipeline.sample(image=image)
            if fig.is_rasterizable:
                code = fig.code
                logger.info("DeTikZify sample: generated %d chars of TikZ code", len(code))
                return code
            else:
                logger.warning("DeTikZify sample: generated code is not rasterizable")
                return str(fig.code) if hasattr(fig, "code") else None
        except Exception as e:
            logger.error("DeTikZify sample failed: %s", e)
            return None

    def simulate(
        self,
        image: Any,
        timeout: int = 300,
        top_k: int = 3,
    ) -> List[Tuple[float, str]]:
        """Run MCTS-based inference to generate and iteratively refine TikZ code.

        Args:
            image: PIL Image, file path, or URL.
            timeout: Maximum time in seconds for MCTS search.
            top_k: Number of top results to return.

        Returns:
            List of (score, tikz_code) tuples sorted by score descending.
        """
        if not self._loaded:
            self.load()

        image = self._prepare_image(image)
        logger.info("DeTikZify simulate: MCTS with timeout=%ds", timeout)

        results = set()
        try:
            for score, fig in self._pipeline.simulate(image=image, timeout=timeout):
                results.add((score, fig.code))
        except Exception as e:
            logger.error("DeTikZify simulate failed: %s", e)

        sorted_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k]
        logger.info("DeTikZify simulate: found %d results", len(sorted_results))
        return sorted_results

    def rasterize(self, tikz_code: str) -> Optional[Image.Image]:
        """Compile and rasterize TikZ code to an image.

        Args:
            tikz_code: TikZ code string.

        Returns:
            PIL Image if compilation succeeds, None otherwise.
        """
        if not self._loaded:
            self.load()

        try:
            from detikzify.infer import TikZDocument
            doc = TikZDocument(tikz_code)
            if doc.is_rasterizable:
                return doc.rasterize()
        except Exception as e:
            logger.error("DeTikZify rasterize failed: %s", e)

        return None

    def _prepare_image(self, image: Any) -> Any:
        """Convert image input to the format DeTikZify expects.

        Args:
            image: PIL Image, file path string, or Path.

        Returns:
            Image in appropriate format for DeTikZify pipeline.
        """
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.exists():
                return Image.open(path).convert("RGB")
            # Could be a URL
            return str(image)
        return image
