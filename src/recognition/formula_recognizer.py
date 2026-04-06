"""Math formula recognition: image to LaTeX math code."""

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger("pdf2latex.recognition.formula")


class FormulaRecognizer:
    """Recognize mathematical formulas and convert to LaTeX code.

    Uses pix2tex (LaTeX-OCR) as primary engine, with Ollama VLM as fallback.
    """

    def __init__(self, ollama_client=None, ollama_model: str = "minicpm-v"):
        """Initialize FormulaRecognizer.

        Args:
            ollama_client: OllamaClient instance for fallback recognition.
            ollama_model: Ollama VLM model name for fallback.
        """
        self.ollama_client = ollama_client
        self.ollama_model = ollama_model
        self._pix2tex_model = None

    def _init_pix2tex(self) -> None:
        """Lazy-initialize pix2tex model."""
        if self._pix2tex_model is not None:
            return

        try:
            from pix2tex.cli import LatexOCR
            logger.info("Initializing pix2tex (LaTeX-OCR)")
            self._pix2tex_model = LatexOCR()
        except ImportError:
            logger.warning("pix2tex not installed, formula recognition will use fallback only")

    def recognize(self, image: np.ndarray, use_fallback: bool = True) -> Optional[str]:
        """Recognize a math formula from an image.

        Args:
            image: Cropped formula region as numpy array (BGR).
            use_fallback: Whether to try VLM fallback if pix2tex fails.

        Returns:
            LaTeX math code string, or None if recognition fails.
        """
        # Try pix2tex first
        result = self._recognize_pix2tex(image)
        if result:
            return result

        # Fallback to VLM
        if use_fallback and self.ollama_client:
            result = self._recognize_vlm(image)
            if result:
                return result

        logger.warning("Formula recognition failed for image region")
        return None

    def _recognize_pix2tex(self, image: np.ndarray) -> Optional[str]:
        """Recognize formula using pix2tex.

        Args:
            image: Formula image as BGR numpy array.

        Returns:
            LaTeX math code or None.
        """
        self._init_pix2tex()
        if self._pix2tex_model is None:
            return None

        try:
            # Convert BGR to RGB PIL Image
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            result = self._pix2tex_model(pil_image)
            if result and result.strip():
                logger.debug("pix2tex result: %s", result[:100])
                return result.strip()
        except Exception as e:
            logger.warning("pix2tex recognition failed: %s", e)

        return None

    def _recognize_vlm(self, image: np.ndarray) -> Optional[str]:
        """Recognize formula using Ollama VLM (minicpm-v).

        Passes the numpy array directly to OllamaClient which encodes it
        in-memory as PNG, avoiding disk I/O with temp files.

        Args:
            image: Formula image as BGR numpy array.

        Returns:
            LaTeX math code or None.
        """
        try:
            prompt = (
                "This image contains a mathematical formula. "
                "Please output ONLY the LaTeX math code for this formula, "
                "without any surrounding $ or $$ delimiters, "
                "without any explanation. Just the raw LaTeX math code."
            )

            result = self.ollama_client.generate(
                model=self.ollama_model,
                prompt=prompt,
                images=[image],
                temperature=0.05,
            )

            if result and result.strip():
                cleaned = self._clean_latex_response(result)
                logger.debug("VLM formula result: %s", cleaned[:100])
                return cleaned

        except Exception as e:
            logger.warning("VLM formula recognition failed: %s", e)

        return None

    def format_as_inline(self, latex_code: str) -> str:
        """Format LaTeX math code as inline math.

        Args:
            latex_code: Raw LaTeX math code.

        Returns:
            Inline math formatted string.
        """
        return f"${latex_code}$"

    def format_as_display(self, latex_code: str) -> str:
        """Format LaTeX math code as display math.

        Args:
            latex_code: Raw LaTeX math code.

        Returns:
            Display math formatted string.
        """
        return f"\\[\n{latex_code}\n\\]"

    @staticmethod
    def _clean_latex_response(response: str) -> str:
        """Clean up LLM response to extract pure LaTeX math code.

        Args:
            response: Raw LLM response text.

        Returns:
            Cleaned LaTeX math code.
        """
        text = response.strip()

        # Remove markdown code block wrappers
        if text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()
            if text.startswith("latex"):
                text = text[5:].strip()

        # Remove surrounding $ or $$
        if text.startswith("$$") and text.endswith("$$"):
            text = text[2:-2].strip()
        elif text.startswith("$") and text.endswith("$"):
            text = text[1:-1].strip()

        # Remove \[ \] wrappers
        if text.startswith("\\[") and text.endswith("\\]"):
            text = text[2:-2].strip()

        # Fix common LaTeX syntax errors
        text = FormulaRecognizer._fix_common_latex_errors(text)

        return text

    @staticmethod
    def _fix_common_latex_errors(latex: str) -> str:
        """Fix common LaTeX syntax errors in recognized formulas.

        Args:
            latex: Raw LaTeX code.

        Returns:
            Fixed LaTeX code.
        """
        # Fix common command errors
        fixes = {
            r'\sqr': r'\sqrt',
            r'\sinx': r'\sin x',
            r'\cosx': r'\cos x',
            r'\tanx': r'\tan x',
            r'\lnx': r'\ln x',
            r'\logx': r'\log x',
        }

        for wrong, correct in fixes.items():
            latex = latex.replace(wrong, correct)

        # Fix missing braces around fractions and roots
        import re
        
        # Fix \sqrt without braces
        latex = re.sub(r'\\sqrt\s+([a-zA-Z])', r'\\sqrt{\1}', latex)
        
        # Fix fractions without braces
        latex = re.sub(r'\\frac\s+([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\\frac{\1}{\2}', latex)
        
        # Fix powers without braces
        latex = re.sub(r'\^([a-zA-Z0-9]+)', r'^{\1}', latex)
        
        # Fix subscripts without braces  
        latex = re.sub(r'_([a-zA-Z0-9]+)', r'_{\1}', latex)

        return latex
