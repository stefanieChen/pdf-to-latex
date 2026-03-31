"""Unit tests for layout detection module."""

import numpy as np
import pytest

from src.layout.layout_detector import BBox, DetectedRegion, RegionType


class TestBBox:
    """Tests for BBox dataclass."""

    def test_basic_properties(self):
        """Test basic bbox properties."""
        bbox = BBox(x1=10, y1=20, x2=110, y2=70)
        assert bbox.width == 100
        assert bbox.height == 50
        assert bbox.area == 5000
        assert bbox.center == (60, 45)

    def test_to_tuple(self):
        """Test conversion to tuple."""
        bbox = BBox(x1=0, y1=0, x2=100, y2=200)
        assert bbox.to_tuple() == (0, 0, 100, 200)


class TestDetectedRegion:
    """Tests for DetectedRegion dataclass."""

    def test_sort_key(self):
        """Test reading order sort key."""
        r1 = DetectedRegion(
            region_type=RegionType.TEXT,
            bbox=BBox(10, 100, 200, 150),
            confidence=0.9,
        )
        r2 = DetectedRegion(
            region_type=RegionType.TEXT,
            bbox=BBox(10, 50, 200, 90),
            confidence=0.9,
        )
        # r2 is above r1 (smaller y1), so r2 should come first
        regions = sorted([r1, r2], key=lambda r: r.sort_key)
        assert regions[0] == r2
        assert regions[1] == r1

    def test_region_types(self):
        """Test all region types are valid."""
        for rtype in RegionType:
            r = DetectedRegion(
                region_type=rtype,
                bbox=BBox(0, 0, 100, 100),
                confidence=0.5,
            )
            assert r.region_type == rtype


class TestLayoutDetector:
    """Tests for LayoutDetector (without PaddleOCR dependency)."""

    def test_detect_columns_single(self):
        """Test single column detection."""
        from src.layout.layout_detector import LayoutDetector
        detector = LayoutDetector.__new__(LayoutDetector)
        detector.confidence_threshold = 0.5

        # All regions centered
        regions = [
            DetectedRegion(RegionType.TEXT, BBox(100, 0, 500, 50), 0.9),
            DetectedRegion(RegionType.TEXT, BBox(100, 60, 500, 110), 0.9),
        ]
        assert detector.detect_columns(regions, 600) == 1

    def test_detect_columns_double(self):
        """Test two-column detection."""
        from src.layout.layout_detector import LayoutDetector
        detector = LayoutDetector.__new__(LayoutDetector)
        detector.confidence_threshold = 0.5

        # Regions split left and right
        regions = [
            DetectedRegion(RegionType.TEXT, BBox(10, 0, 250, 50), 0.9),
            DetectedRegion(RegionType.TEXT, BBox(10, 60, 250, 110), 0.9),
            DetectedRegion(RegionType.TEXT, BBox(350, 0, 590, 50), 0.9),
            DetectedRegion(RegionType.TEXT, BBox(350, 60, 590, 110), 0.9),
        ]
        assert detector.detect_columns(regions, 600) == 2

    def test_crop_region(self):
        """Test region cropping."""
        from src.layout.layout_detector import LayoutDetector
        detector = LayoutDetector.__new__(LayoutDetector)

        image = np.zeros((500, 500, 3), dtype=np.uint8)
        region = DetectedRegion(RegionType.TEXT, BBox(50, 50, 200, 150), 0.9)

        cropped = detector.crop_region(image, region, padding=0)
        assert cropped.shape == (100, 150, 3)

    def test_crop_region_with_padding(self):
        """Test region cropping with padding."""
        from src.layout.layout_detector import LayoutDetector
        detector = LayoutDetector.__new__(LayoutDetector)

        image = np.zeros((500, 500, 3), dtype=np.uint8)
        region = DetectedRegion(RegionType.TEXT, BBox(50, 50, 200, 150), 0.9)

        cropped = detector.crop_region(image, region, padding=10)
        assert cropped.shape == (110 + 10, 160 + 10, 3)  # Padded dimensions
