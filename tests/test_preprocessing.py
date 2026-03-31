"""Unit tests for preprocessing modules."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.batch_loader import BatchLoader, SUPPORTED_EXTS


class TestImageEnhancer:
    """Tests for ImageEnhancer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.enhancer = ImageEnhancer()
        # Create a simple test image (white background with some noise)
        self.test_image = np.ones((200, 300, 3), dtype=np.uint8) * 255
        # Add some noise
        noise = np.random.randint(0, 50, self.test_image.shape, dtype=np.uint8)
        self.test_image = cv2.subtract(self.test_image, noise)

    def test_denoise(self):
        """Test denoising produces output of same shape."""
        result = self.enhancer.apply_denoise(self.test_image)
        assert result.shape == self.test_image.shape

    def test_binarize(self):
        """Test binarization produces binary output."""
        result = self.enhancer.apply_binarize(self.test_image)
        unique_values = np.unique(result)
        assert all(v in [0, 255] for v in unique_values)

    def test_detect_skew_angle(self):
        """Test skew angle detection returns a float."""
        angle = self.enhancer.detect_skew_angle(self.test_image)
        assert isinstance(angle, float)
        assert -45 <= angle <= 45

    def test_deskew(self):
        """Test deskewing produces valid output."""
        result = self.enhancer.apply_deskew(self.test_image, 2.0)
        assert result is not None
        assert len(result.shape) == 3

    def test_full_enhance_pipeline(self):
        """Test the full enhancement pipeline."""
        result = self.enhancer.enhance(
            self.test_image,
            denoise=True,
            binarize=False,
            deskew=True,
        )
        assert result is not None
        assert len(result.shape) >= 2

    def test_enhance_file(self):
        """Test enhancing an image file."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, self.test_image)
            tmp_path = Path(f.name)

        try:
            output = Path(tempfile.mktemp(suffix=".png"))
            result_path = self.enhancer.enhance_file(tmp_path, output)
            assert result_path.exists()
            output.unlink(missing_ok=True)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestBatchLoader:
    """Tests for BatchLoader."""

    def setup_method(self):
        """Set up test fixtures."""
        self.loader = BatchLoader()

    def test_default_supported_formats(self):
        """Test default supported formats include common types."""
        assert ".pdf" in self.loader.supported_formats
        assert ".png" in self.loader.supported_formats
        assert ".jpg" in self.loader.supported_formats

    def test_custom_formats(self):
        """Test custom format specification."""
        loader = BatchLoader(supported_formats=[".png", ".pdf"])
        assert loader.supported_formats == {".png", ".pdf"}

    def test_validate_file_nonexistent(self):
        """Test validation of nonexistent file."""
        assert not self.loader.validate_file(Path("/nonexistent/file.png"))

    def test_validate_file_unsupported(self):
        """Test validation of unsupported file type."""
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            assert not self.loader.validate_file(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_scan_directory(self):
        """Test scanning a directory for supported files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "test.png").write_bytes(b"fake")
            (tmp / "test.txt").write_bytes(b"fake")
            (tmp / "test.jpg").write_bytes(b"fake")

            files = self.loader.scan_directory(tmp)
            names = [f.name for f in files]
            assert "test.png" in names
            assert "test.jpg" in names
            assert "test.txt" not in names

    def test_classify_files(self):
        """Test file classification into PDFs and images."""
        files = [Path("a.pdf"), Path("b.png"), Path("c.jpg"), Path("d.pdf")]
        pdfs, images = self.loader.classify_files(files)
        assert len(pdfs) == 2
        assert len(images) == 2

    def test_load_image(self):
        """Test loading a valid image file."""
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, img)
            tmp_path = Path(f.name)
        try:
            loaded = self.loader.load_image(tmp_path)
            assert loaded.shape == (100, 100, 3)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_load_image_invalid(self):
        """Test loading an invalid image raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not an image")
            tmp_path = Path(f.name)
        try:
            with pytest.raises(ValueError):
                self.loader.load_image(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
