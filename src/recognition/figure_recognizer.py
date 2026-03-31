"""Figure/geometry recognition: image to TikZ code via DeTikZify."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("pdf2latex.recognition.figure")


class FigureRecognizer:
    """Convert figure/geometry images to TikZ code.

    Three-tier strategy:
    1. DeTikZify (primary) — specialized model with MCTS optimization
    2. Ollama VLM (fallback) — general vision model
    3. includegraphics (final fallback) — embed original image
    """

    def __init__(
        self,
        detikzify_client=None,
        ollama_client=None,
        model_scheduler=None,
        ollama_vlm_model: str = "minicpm-v",
        mcts_timeout: int = 300,
    ):
        """Initialize FigureRecognizer.

        Args:
            detikzify_client: DeTikZifyClient instance.
            ollama_client: OllamaClient instance for VLM fallback.
            model_scheduler: ModelScheduler for VRAM management.
            ollama_vlm_model: Ollama VLM model name.
            mcts_timeout: Timeout in seconds for DeTikZify MCTS.
        """
        self.detikzify = detikzify_client
        self.ollama = ollama_client
        self.scheduler = model_scheduler
        self.ollama_vlm_model = ollama_vlm_model
        self.mcts_timeout = mcts_timeout

    def recognize(
        self,
        image: np.ndarray,
        use_mcts: bool = False,
        save_original_path: Optional[Path] = None,
    ) -> str:
        """Recognize a figure and return LaTeX code.

        Args:
            image: Cropped figure region as numpy array (BGR).
            use_mcts: Whether to use MCTS for higher quality (slower).
            save_original_path: If set, save original image here for includegraphics fallback.

        Returns:
            LaTeX code: either TikZ environment or \\includegraphics command.
        """
        # Try DeTikZify first
        tikz_code = self._try_detikzify(image, use_mcts=use_mcts)
        if tikz_code:
            return tikz_code

        # Try Ollama VLM fallback
        tikz_code = self._try_vlm(image)
        if tikz_code:
            return tikz_code

        # Final fallback: includegraphics
        return self._fallback_includegraphics(image, save_original_path)

    def _try_detikzify(self, image: np.ndarray, use_mcts: bool = False) -> Optional[str]:
        """Try to convert figure using DeTikZify.

        Args:
            image: Figure image (BGR).
            use_mcts: Whether to use MCTS inference.

        Returns:
            TikZ code or None.
        """
        if self.detikzify is None:
            return None

        try:
            from src.model_scheduler import ModelType

            # Acquire DeTikZify in scheduler
            if self.scheduler:
                self.scheduler.acquire(ModelType.DETIKZIFY)

            pil_image = self._bgr_to_pil(image)

            if use_mcts:
                results = self.detikzify.simulate(
                    image=pil_image,
                    timeout=self.mcts_timeout,
                    top_k=1,
                )
                if results:
                    score, code = results[0]
                    logger.info("DeTikZify MCTS: score=%.3f, code_len=%d", score, len(code))
                    return code
            else:
                code = self.detikzify.sample(image=pil_image)
                if code:
                    logger.info("DeTikZify sample: code_len=%d", len(code))
                    return code

        except Exception as e:
            logger.warning("DeTikZify recognition failed: %s", e)

        return None

    def _try_vlm(self, image: np.ndarray) -> Optional[str]:
        """Try to convert figure using Ollama VLM.

        Args:
            image: Figure image (BGR).

        Returns:
            TikZ code or None.
        """
        if self.ollama is None:
            return None

        try:
            from src.model_scheduler import ModelType

            if self.scheduler:
                self.scheduler.acquire(ModelType.OLLAMA_VLM)

            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                tmp_path = tmp.name

            prompt = (
                "Convert this figure/diagram into TikZ code for LaTeX. "
                "Output ONLY the complete TikZ code starting with \\begin{tikzpicture} "
                "and ending with \\end{tikzpicture}. "
                "Do not include any explanation or surrounding text."
            )

            result = self.ollama.generate(
                model=self.ollama_vlm_model,
                prompt=prompt,
                images=[tmp_path],
                temperature=0.1,
            )

            Path(tmp_path).unlink(missing_ok=True)

            if result and "\\begin{tikzpicture}" in result:
                code = self._extract_tikz(result)
                if code:
                    logger.info("VLM TikZ result: code_len=%d", len(code))
                    return code

        except Exception as e:
            logger.warning("VLM figure recognition failed: %s", e)

        return None

    def _fallback_includegraphics(
        self,
        image: np.ndarray,
        save_path: Optional[Path] = None,
    ) -> str:
        """Generate \\includegraphics command as final fallback.

        Args:
            image: Figure image (BGR).
            save_path: Path to save the image file.

        Returns:
            LaTeX \\includegraphics command.
        """
        if save_path is None:
            save_path = Path(tempfile.mktemp(suffix=".png", prefix="fig_"))

        save_path = Path(save_path)
        cv2.imwrite(str(save_path), image)
        logger.info("Figure fallback: saved image to %s", save_path.name)

        # Use relative path in LaTeX
        return (
            f"\\begin{{figure}}[H]\n"
            f"\\centering\n"
            f"\\includegraphics[width=0.8\\textwidth]{{{save_path.name}}}\n"
            f"\\end{{figure}}"
        )

    @staticmethod
    def _bgr_to_pil(image: np.ndarray) -> Image.Image:
        """Convert BGR numpy array to RGB PIL Image.

        Args:
            image: BGR numpy array.

        Returns:
            PIL Image in RGB mode.
        """
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _extract_tikz(text: str) -> Optional[str]:
        """Extract TikZ code block from LLM response.

        Args:
            text: Raw LLM response.

        Returns:
            Extracted TikZ code or None.
        """
        import re
        # Try to find tikzpicture environment
        pattern = r"(\\begin\{tikzpicture\}.*?\\end\{tikzpicture\})"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        return None
