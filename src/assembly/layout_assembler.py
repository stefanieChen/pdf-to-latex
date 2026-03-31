"""Assemble recognized regions into a complete LaTeX document."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.layout.layout_detector import DetectedRegion, RegionType

logger = logging.getLogger("pdf2latex.assembly.assembler")


@dataclass
class PageContent:
    """Content of a single page ready for assembly."""
    page_num: int
    regions: List[Dict[str, Any]]
    num_columns: int = 1
    width: int = 0
    height: int = 0


class LayoutAssembler:
    """Assemble recognized content into LaTeX body text following original layout."""

    def __init__(self):
        """Initialize LayoutAssembler."""
        pass

    def assemble_page(self, recognized_regions: List[Dict[str, Any]], num_columns: int = 1) -> str:
        """Assemble recognized regions of a single page into LaTeX.

        Args:
            recognized_regions: List of dicts with keys:
                - 'type': RegionType
                - 'latex': LaTeX code for the region
                - 'bbox': (x1, y1, x2, y2) bounding box
            num_columns: Detected number of text columns.

        Returns:
            LaTeX body content for this page.
        """
        if not recognized_regions:
            return ""

        # Sort by reading order: top-to-bottom, left-to-right
        sorted_regions = sorted(recognized_regions, key=lambda r: (r["bbox"][1], r["bbox"][0]))

        parts = []

        if num_columns > 1:
            parts.append(f"\\begin{{multicols}}{{{num_columns}}}")

        for region in sorted_regions:
            latex = region.get("latex", "")
            region_type = region.get("type", RegionType.TEXT)

            if not latex.strip():
                continue

            formatted = self._format_region(latex, region_type)
            parts.append(formatted)

        if num_columns > 1:
            parts.append("\\end{multicols}")

        return "\n\n".join(parts)

    def assemble_document(
        self,
        pages: List[Dict[str, Any]],
        template_manager=None,
        title: Optional[str] = None,
    ) -> str:
        """Assemble multiple pages into a complete LaTeX document.

        Args:
            pages: List of page dicts, each containing:
                - 'regions': list of recognized region dicts
                - 'num_columns': detected column count
            template_manager: TemplateManager instance for document wrapping.
            title: Optional document title.

        Returns:
            Complete LaTeX document string.
        """
        page_contents = []

        for i, page in enumerate(pages):
            regions = page.get("regions", [])
            num_cols = page.get("num_columns", 1)

            page_latex = self.assemble_page(regions, num_cols)
            if page_latex.strip():
                page_contents.append(page_latex)

        body = "\n\n\\newpage\n\n".join(page_contents)

        if template_manager:
            return template_manager.wrap_document(body, title=title)
        return body

    def _format_region(self, latex: str, region_type: RegionType) -> str:
        """Format a region's LaTeX code based on its type.

        Args:
            latex: Raw LaTeX code for the region.
            region_type: Type of the region.

        Returns:
            Formatted LaTeX code.
        """
        if region_type == RegionType.TITLE:
            return f"\\section*{{{latex.strip()}}}"

        elif region_type == RegionType.FORMULA:
            # If it already has math delimiters, use as-is
            if any(d in latex for d in ["\\[", "\\begin{equation}", "\\begin{align}"]):
                return latex
            # Wrap as display math
            return f"\\[\n{latex.strip()}\n\\]"

        elif region_type == RegionType.TABLE:
            # Tables should already be properly formatted
            return latex

        elif region_type == RegionType.FIGURE:
            # Figures should already be formatted (TikZ or includegraphics)
            return latex

        else:
            # Plain text paragraph
            return latex.strip()
