"""Batch loader for image files and PDF documents."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.preprocessing.loader")

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS


class BatchLoader:
    """Load and organize input files (images and PDFs) for processing."""

    def __init__(self, supported_formats: Optional[List[str]] = None):
        """Initialize BatchLoader.

        Args:
            supported_formats: List of supported file extensions (with dot).
                Defaults to common image formats plus PDF.
        """
        if supported_formats:
            self.supported_formats = {ext.lower() for ext in supported_formats}
        else:
            self.supported_formats = SUPPORTED_EXTS

    def scan_directory(self, directory: Path) -> List[Path]:
        """Scan a directory for supported files, sorted by name.

        Args:
            directory: Directory to scan.

        Returns:
            Sorted list of supported file paths.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        files = []
        for f in sorted(directory.iterdir()):
            if f.is_file() and f.suffix.lower() in self.supported_formats:
                files.append(f)

        logger.info("Found %d supported files in %s", len(files), directory)
        return files

    def classify_files(self, files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """Classify files into PDFs and images.

        Args:
            files: List of file paths.

        Returns:
            Tuple of (pdf_files, image_files).
        """
        pdfs = [f for f in files if f.suffix.lower() in SUPPORTED_PDF_EXTS]
        images = [f for f in files if f.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        logger.info("Classified: %d PDFs, %d images", len(pdfs), len(images))
        return pdfs, images

    def load_image(self, image_path: Path) -> np.ndarray:
        """Load a single image as numpy array (BGR).

        Args:
            image_path: Path to image file.

        Returns:
            Image as numpy array in BGR format.

        Raises:
            ValueError: If image cannot be read.
        """
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return image

    def load_images(self, image_paths: List[Path]) -> List[Tuple[Path, np.ndarray]]:
        """Load multiple images.

        Args:
            image_paths: List of image file paths.

        Returns:
            List of (path, image_array) tuples for successfully loaded images.
        """
        results = []
        for path in image_paths:
            try:
                img = self.load_image(path)
                results.append((path, img))
            except ValueError as e:
                logger.warning("Skipping unreadable image: %s (%s)", path, e)
        return results

    def validate_file(self, file_path: Path) -> bool:
        """Check if a file is supported and readable.

        Args:
            file_path: Path to file.

        Returns:
            True if file is valid and supported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False
        if file_path.suffix.lower() not in self.supported_formats:
            return False
        return True
