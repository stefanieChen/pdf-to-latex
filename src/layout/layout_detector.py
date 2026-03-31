"""Document layout detection using PaddleOCR PPStructure."""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.layout")


class RegionType(Enum):
    """Types of detected document regions."""
    TEXT = "text"
    TITLE = "title"
    FORMULA = "formula"
    TABLE = "table"
    FIGURE = "figure"
    UNKNOWN = "unknown"


@dataclass
class BBox:
    """Bounding box with (x1, y1, x2, y2) coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Width of the bounding box."""
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        """Height of the bounding box."""
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        """Area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Center point of the bounding box."""
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class DetectedRegion:
    """A detected region in a document page."""
    region_type: RegionType
    bbox: BBox
    confidence: float
    page_num: int = 0
    content: Optional[str] = None
    raw_data: dict = field(default_factory=dict)

    @property
    def sort_key(self) -> Tuple[int, int]:
        """Key for sorting by reading order (top-to-bottom, left-to-right)."""
        return (self.bbox.y1, self.bbox.x1)


class LayoutDetector:
    """Detect document layout regions using PaddleOCR PPStructure."""

    # Map PPStructure type strings to RegionType
    TYPE_MAP = {
        "text": RegionType.TEXT,
        "title": RegionType.TITLE,
        "figure": RegionType.FIGURE,
        "figure_caption": RegionType.TEXT,
        "table": RegionType.TABLE,
        "table_caption": RegionType.TEXT,
        "header": RegionType.TEXT,
        "footer": RegionType.TEXT,
        "reference": RegionType.TEXT,
        "equation": RegionType.FORMULA,
    }

    def __init__(self, confidence_threshold: float = 0.5, use_gpu: bool = True):
        """Initialize LayoutDetector.

        Args:
            confidence_threshold: Minimum confidence to accept a detection.
            use_gpu: Whether to use GPU for inference.
        """
        self.confidence_threshold = confidence_threshold
        self.use_gpu = use_gpu
        self._engine = None

    def _init_engine(self) -> None:
        """Lazy-initialize the PPStructure engine."""
        if self._engine is not None:
            return

        from paddleocr import PPStructure

        logger.info("Initializing PPStructure layout engine (GPU=%s)", self.use_gpu)
        self._engine = PPStructure(
            table=False,
            ocr=False,
            show_log=False,
            use_gpu=self.use_gpu,
            layout=True,
        )

    def detect(self, image: np.ndarray, page_num: int = 0) -> List[DetectedRegion]:
        """Detect layout regions in a document image.

        Args:
            image: Input image as numpy array (BGR).
            page_num: Page number for tracking.

        Returns:
            List of DetectedRegion sorted by reading order.
        """
        self._init_engine()

        logger.debug("Detecting layout for page %d (image: %dx%d)", page_num, image.shape[1], image.shape[0])

        results = self._engine(image)
        regions = []

        for item in results:
            region_type_str = item.get("type", "unknown").lower()
            region_type = self.TYPE_MAP.get(region_type_str, RegionType.UNKNOWN)

            bbox_coords = item.get("bbox", [0, 0, 0, 0])
            if len(bbox_coords) == 4:
                bbox = BBox(
                    x1=int(bbox_coords[0]),
                    y1=int(bbox_coords[1]),
                    x2=int(bbox_coords[2]),
                    y2=int(bbox_coords[3]),
                )
            else:
                continue

            confidence = float(item.get("score", 0.0))

            if confidence < self.confidence_threshold:
                continue

            region = DetectedRegion(
                region_type=region_type,
                bbox=bbox,
                confidence=confidence,
                page_num=page_num,
                raw_data=item,
            )
            regions.append(region)

        # Sort by reading order
        regions.sort(key=lambda r: r.sort_key)
        logger.info("Page %d: detected %d regions", page_num, len(regions))
        return regions

    def detect_from_file(self, image_path: Path, page_num: int = 0) -> List[DetectedRegion]:
        """Detect layout from an image file.

        Args:
            image_path: Path to image file.
            page_num: Page number for tracking.

        Returns:
            List of DetectedRegion.
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return self.detect(image, page_num)

    def crop_region(self, image: np.ndarray, region: DetectedRegion, padding: int = 5) -> np.ndarray:
        """Crop a detected region from the image with optional padding.

        Args:
            image: Full page image.
            region: Detected region to crop.
            padding: Pixels of padding around the region.

        Returns:
            Cropped image of the region.
        """
        h, w = image.shape[:2]
        x1 = max(0, region.bbox.x1 - padding)
        y1 = max(0, region.bbox.y1 - padding)
        x2 = min(w, region.bbox.x2 + padding)
        y2 = min(h, region.bbox.y2 + padding)
        return image[y1:y2, x1:x2].copy()

    def detect_columns(self, regions: List[DetectedRegion], page_width: int) -> int:
        """Estimate the number of text columns on a page.

        Args:
            regions: List of detected regions.
            page_width: Width of the page image in pixels.

        Returns:
            Estimated number of columns (1 or 2).
        """
        text_regions = [r for r in regions if r.region_type in (RegionType.TEXT, RegionType.TITLE)]
        if not text_regions:
            return 1

        centers_x = [r.bbox.center[0] for r in text_regions]
        mid = page_width // 2

        left_count = sum(1 for cx in centers_x if cx < mid * 0.8)
        right_count = sum(1 for cx in centers_x if cx > mid * 1.2)

        if left_count >= 2 and right_count >= 2:
            return 2
        return 1
