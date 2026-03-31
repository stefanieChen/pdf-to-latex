"""Unit tests for validation modules."""

import numpy as np
import pytest

from src.validation.visual_comparator import VisualComparator
from src.validation.llm_reviewer import LLMReviewer


class TestVisualComparator:
    """Tests for VisualComparator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.comparator = VisualComparator(ssim_threshold=0.85)

    def test_identical_images(self):
        """Test SSIM of identical images is ~1.0."""
        img = np.random.randint(0, 255, (200, 300), dtype=np.uint8)
        score, passes = self.comparator.compare(img, img.copy())
        assert score > 0.99
        assert passes is True

    def test_different_images(self):
        """Test SSIM of very different images is low."""
        img1 = np.zeros((200, 300), dtype=np.uint8)
        img2 = np.ones((200, 300), dtype=np.uint8) * 255
        score, passes = self.comparator.compare(img1, img2)
        assert score < 0.1
        assert passes is False

    def test_slightly_different_images(self):
        """Test SSIM with slight noise."""
        img1 = np.ones((200, 300), dtype=np.uint8) * 128
        img2 = img1.copy()
        noise = np.random.randint(-10, 10, img2.shape, dtype=np.int16)
        img2 = np.clip(img2.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        score, _ = self.comparator.compare(img1, img2)
        assert 0.5 < score < 1.0

    def test_different_size_images(self):
        """Test comparison of differently sized images."""
        img1 = np.ones((200, 300), dtype=np.uint8) * 128
        img2 = np.ones((250, 350), dtype=np.uint8) * 128
        score, passes = self.comparator.compare(img1, img2)
        assert score > 0.9  # Same content, different size

    def test_bgr_input(self):
        """Test comparison with BGR input images."""
        img = np.ones((200, 300, 3), dtype=np.uint8) * 128
        score, passes = self.comparator.compare(img, img.copy())
        assert score > 0.99


class TestLLMReviewer:
    """Tests for LLMReviewer (unit-level, no Ollama required)."""

    def test_extract_latex_code_block(self):
        """Test extracting LaTeX from code block."""
        response = '```latex\n\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n```'
        result = LLMReviewer._extract_latex_code(response)
        assert "\\documentclass" in result
        assert "\\begin{document}" in result

    def test_extract_latex_plain(self):
        """Test extracting plain LaTeX."""
        response = "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}"
        result = LLMReviewer._extract_latex_code(response)
        assert "\\documentclass" in result

    def test_extract_latex_no_code(self):
        """Test returning None when no LaTeX found."""
        result = LLMReviewer._extract_latex_code("Just some text without LaTeX")
        assert result is None

    def test_review_without_ollama(self):
        """Test review gracefully handles missing Ollama client."""
        reviewer = LLMReviewer(ollama_client=None)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = reviewer.review(img, img)
        assert result["passed"] is True

    def test_fix_without_ollama(self):
        """Test fix_latex gracefully handles missing Ollama client."""
        reviewer = LLMReviewer(ollama_client=None)
        result = reviewer.fix_latex("bad code", "error log")
        assert result is None
