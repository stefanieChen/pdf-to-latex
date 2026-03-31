"""Table structure recognition and HTML-to-LaTeX conversion."""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.recognition.table")


class TableRecognizer:
    """Recognize table structure using PaddleOCR PPStructure and convert to LaTeX."""

    def __init__(self, use_gpu: bool = True):
        """Initialize TableRecognizer.

        Args:
            use_gpu: Whether to use GPU for inference.
        """
        self.use_gpu = use_gpu
        self._engine = None

    def _init_engine(self) -> None:
        """Lazy-initialize PPStructure table engine."""
        if self._engine is not None:
            return

        from paddleocr import PPStructure

        logger.info("Initializing PPStructure table engine (GPU=%s)", self.use_gpu)
        self._engine = PPStructure(
            table=True,
            ocr=True,
            show_log=False,
            use_gpu=self.use_gpu,
            layout=False,
        )

    def recognize(self, image: np.ndarray) -> Optional[str]:
        """Recognize table structure and return LaTeX code.

        Args:
            image: Cropped table region as numpy array (BGR).

        Returns:
            LaTeX table code string, or None if recognition fails.
        """
        html = self.recognize_to_html(image)
        if html:
            return self.html_to_latex(html)
        return None

    def recognize_to_html(self, image: np.ndarray) -> Optional[str]:
        """Recognize table and return HTML representation.

        Args:
            image: Cropped table region as numpy array (BGR).

        Returns:
            HTML table string, or None if recognition fails.
        """
        self._init_engine()

        try:
            results = self._engine(image)
            for item in results:
                if item.get("type") == "table":
                    html = item.get("res", {}).get("html", "")
                    if html:
                        logger.debug("Table HTML recognized (%d chars)", len(html))
                        return html
        except Exception as e:
            logger.warning("Table recognition failed: %s", e)

        return None

    def html_to_latex(self, html: str) -> str:
        """Convert HTML table to LaTeX tabular environment.

        Args:
            html: HTML table string from PPStructure.

        Returns:
            LaTeX table code.
        """
        rows = self._parse_html_table(html)
        if not rows:
            return ""

        max_cols = max(len(row) for row in rows)

        # Build column specification
        col_spec = "|".join(["c"] * max_cols)
        col_spec = f"|{col_spec}|"

        lines = []
        lines.append("\\begin{table}[H]")
        lines.append("\\centering")
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\hline")

        for i, row in enumerate(rows):
            # Pad row to max_cols
            while len(row) < max_cols:
                row.append("")

            # Escape LaTeX special chars in cell content
            escaped_cells = [self._escape_cell(cell) for cell in row]
            line = " & ".join(escaped_cells) + " \\\\"
            lines.append(line)
            lines.append("\\hline")

        lines.append("\\end{tabular}")
        lines.append("\\end{table}")

        latex = "\n".join(lines)
        logger.debug("Generated LaTeX table (%d rows, %d cols)", len(rows), max_cols)
        return latex

    def _parse_html_table(self, html: str) -> List[List[str]]:
        """Parse HTML table into a 2D list of cell contents.

        Args:
            html: HTML table string.

        Returns:
            2D list where each inner list is a row of cell contents.
        """
        rows = []

        # Find all rows
        row_pattern = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
        cell_pattern = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
        tag_pattern = re.compile(r"<[^>]+>")

        for row_match in row_pattern.finditer(html):
            row_html = row_match.group(1)
            cells = []
            for cell_match in cell_pattern.finditer(row_html):
                cell_content = cell_match.group(1)
                # Strip HTML tags from cell content
                cell_text = tag_pattern.sub("", cell_content).strip()
                cells.append(cell_text)
            if cells:
                rows.append(cells)

        return rows

    @staticmethod
    def _escape_cell(text: str) -> str:
        """Escape LaTeX special characters in table cell content.

        Args:
            text: Raw cell text.

        Returns:
            LaTeX-safe cell text.
        """
        special_chars = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        return text
