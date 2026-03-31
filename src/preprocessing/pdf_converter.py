"""Convert PDF pages to high-resolution images using PyMuPDF."""

import logging
from pathlib import Path
from typing import List, Optional

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

logger = logging.getLogger("pdf2latex.preprocessing.pdf")


class PdfConverter:
    """Convert PDF files to per-page images at configurable DPI."""

    def __init__(self, dpi: int = 300):
        """Initialize PdfConverter.

        Args:
            dpi: Resolution for rendering PDF pages. Default 300.
        """
        self.dpi = dpi
        self._zoom = dpi / 72.0  # PDF default is 72 DPI

    def convert(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
        page_range: Optional[range] = None,
    ) -> List[Path]:
        """Convert PDF pages to PNG images.

        Args:
            pdf_path: Path to input PDF file.
            output_dir: Directory to save images. If None, uses pdf_path parent.
            page_range: Optional range of pages to convert (0-indexed).

        Returns:
            List of paths to generated image files.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if output_dir is None:
            output_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        logger.info("Converting PDF: %s (%d pages, %d DPI)", pdf_path.name, total_pages, self.dpi)

        if page_range is None:
            page_range = range(total_pages)

        image_paths = []
        mat = fitz.Matrix(self._zoom, self._zoom)

        for page_num in page_range:
            if page_num >= total_pages:
                logger.warning("Page %d out of range (total: %d), skipping", page_num, total_pages)
                continue

            page = doc[page_num]
            pix = page.get_pixmap(matrix=mat, alpha=False)

            img_path = output_dir / f"page_{page_num + 1:04d}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)

            logger.debug("Converted page %d/%d -> %s", page_num + 1, total_pages, img_path.name)

        doc.close()
        logger.info("Converted %d pages to %s", len(image_paths), output_dir)
        return image_paths

    def get_page_count(self, pdf_path: Path) -> int:
        """Get the number of pages in a PDF.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Number of pages.
        """
        doc = fitz.open(str(pdf_path))
        count = len(doc)
        doc.close()
        return count

    def convert_page_to_numpy(self, pdf_path: Path, page_num: int = 0) -> np.ndarray:
        """Convert a single PDF page to a numpy array (BGR format for OpenCV).

        Args:
            pdf_path: Path to PDF file.
            page_num: 0-indexed page number.

        Returns:
            Numpy array of the page image in BGR format.
        """
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(self._zoom, self._zoom)
        page = doc[page_num]
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
        # PyMuPDF outputs RGB, convert to BGR for OpenCV compatibility
        img_bgr = img[:, :, ::-1].copy()

        doc.close()
        return img_bgr
