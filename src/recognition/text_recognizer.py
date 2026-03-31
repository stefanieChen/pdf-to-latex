"""Chinese/English text recognition using PaddleOCR."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.recognition.text")


@dataclass
class TextLine:
    """A recognized line of text with position information."""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    polygon: Optional[List[List[int]]] = None

    @property
    def sort_key(self) -> Tuple[int, int]:
        """Key for sorting by reading order."""
        return (self.bbox[1], self.bbox[0])


class TextRecognizer:
    """OCR engine for Chinese/English text using PaddleOCR."""

    def __init__(self, lang: str = "ch", use_angle_cls: bool = True, use_gpu: bool = True):
        """Initialize TextRecognizer.

        Args:
            lang: Language for OCR. 'ch' for Chinese+English, 'en' for English only.
            use_angle_cls: Whether to use text direction classification.
            use_gpu: Whether to use GPU for inference.
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._engine = None

    def _init_engine(self) -> None:
        """Lazy-initialize the PaddleOCR engine."""
        if self._engine is not None:
            return

        from paddleocr import PaddleOCR

        logger.info("Initializing PaddleOCR (lang=%s, angle_cls=%s, GPU=%s)",
                     self.lang, self.use_angle_cls, self.use_gpu)
        self._engine = PaddleOCR(
            use_angle_cls=self.use_angle_cls,
            lang=self.lang,
            use_gpu=self.use_gpu,
            show_log=False,
        )

    def recognize(self, image: np.ndarray) -> List[TextLine]:
        """Recognize text in an image region.

        Args:
            image: Input image as numpy array (BGR).

        Returns:
            List of TextLine objects sorted by reading order.
        """
        self._init_engine()

        results = self._engine.ocr(image, cls=self.use_angle_cls)
        if not results or results[0] is None:
            return []

        lines = []
        for result in results[0]:
            polygon = result[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = result[1][0]
            confidence = result[1][1]

            # Convert polygon to bounding box
            xs = [p[0] for p in polygon]
            ys = [p[1] for p in polygon]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

            lines.append(TextLine(
                text=text,
                confidence=confidence,
                bbox=bbox,
                polygon=[[int(x), int(y)] for x, y in polygon],
            ))

        lines.sort(key=lambda l: l.sort_key)
        logger.debug("Recognized %d text lines", len(lines))
        return lines

    def recognize_to_latex(self, image: np.ndarray) -> str:
        """Recognize text and format as LaTeX paragraphs.

        Args:
            image: Input image as numpy array (BGR).

        Returns:
            LaTeX-formatted text string.
        """
        lines = self.recognize(image)
        if not lines:
            return ""

        # Group lines into paragraphs by vertical proximity
        paragraphs = self._group_paragraphs(lines)
        latex_parts = []

        for paragraph in paragraphs:
            text = " ".join(line.text for line in paragraph)
            # Escape special LaTeX characters
            text = self._escape_latex(text)
            latex_parts.append(text)

        return "\n\n".join(latex_parts)

    def _group_paragraphs(self, lines: List[TextLine], line_gap_ratio: float = 1.5) -> List[List[TextLine]]:
        """Group text lines into paragraphs based on vertical spacing.

        Args:
            lines: Sorted list of text lines.
            line_gap_ratio: Gap threshold as ratio of average line height.

        Returns:
            List of paragraph groups.
        """
        if not lines:
            return []

        avg_height = np.mean([l.bbox[3] - l.bbox[1] for l in lines])
        gap_threshold = avg_height * line_gap_ratio

        paragraphs = [[lines[0]]]
        for i in range(1, len(lines)):
            prev_bottom = lines[i - 1].bbox[3]
            curr_top = lines[i].bbox[1]
            gap = curr_top - prev_bottom

            if gap > gap_threshold:
                paragraphs.append([lines[i]])
            else:
                paragraphs[-1].append(lines[i])

        return paragraphs

    @staticmethod
    def _escape_latex(text: str) -> str:
        """Escape special LaTeX characters in text.

        Args:
            text: Raw text string.

        Returns:
            LaTeX-safe text string.
        """
        special_chars = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
            "~": r"\textasciitilde{}",
            "^": r"\textasciicircum{}",
        }
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        return text
