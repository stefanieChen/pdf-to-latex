"""Unit tests for assembly modules."""

import pytest

from src.assembly.template_manager import TemplateManager
from src.layout.layout_detector import RegionType


class TestTemplateManager:
    """Tests for TemplateManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tm = TemplateManager()

    def test_generate_preamble(self):
        """Test preamble generation includes required packages."""
        preamble = self.tm.generate_preamble()
        assert "\\documentclass" in preamble
        assert "ctex" in preamble
        assert "amsmath" in preamble
        assert "tikz" in preamble
        assert "graphicx" in preamble

    def test_wrap_document(self):
        """Test wrapping body in a complete document."""
        body = "Hello, world!"
        doc = self.tm.wrap_document(body)
        assert "\\begin{document}" in doc
        assert "\\end{document}" in doc
        assert "Hello, world!" in doc

    def test_wrap_document_with_title(self):
        """Test wrapping document with title."""
        doc = self.tm.wrap_document("Content", title="Test Title")
        assert "\\title{Test Title}" in doc
        assert "\\maketitle" in doc


class TestLayoutAssembler:
    """Tests for LayoutAssembler."""

    def setup_method(self):
        """Set up test fixtures."""
        from src.assembly.layout_assembler import LayoutAssembler
        self.assembler = LayoutAssembler()

    def test_assemble_empty_page(self):
        """Test assembling an empty page."""
        result = self.assembler.assemble_page([])
        assert result == ""

    def test_assemble_text_regions(self):
        """Test assembling text regions."""
        regions = [
            {"type": RegionType.TEXT, "latex": "First paragraph.", "bbox": (0, 0, 100, 30)},
            {"type": RegionType.TEXT, "latex": "Second paragraph.", "bbox": (0, 40, 100, 70)},
        ]
        result = self.assembler.assemble_page(regions)
        assert "First paragraph." in result
        assert "Second paragraph." in result

    def test_assemble_title_region(self):
        """Test assembling a title region."""
        regions = [
            {"type": RegionType.TITLE, "latex": "My Title", "bbox": (0, 0, 200, 40)},
        ]
        result = self.assembler.assemble_page(regions)
        assert "\\section*{My Title}" in result

    def test_assemble_formula_region(self):
        """Test assembling a formula region."""
        regions = [
            {"type": RegionType.FORMULA, "latex": "E = mc^2", "bbox": (0, 0, 100, 30)},
        ]
        result = self.assembler.assemble_page(regions)
        assert "\\[" in result
        assert "E = mc^2" in result
        assert "\\]" in result

    def test_assemble_multicol(self):
        """Test multi-column assembly."""
        regions = [
            {"type": RegionType.TEXT, "latex": "Column content", "bbox": (0, 0, 100, 30)},
        ]
        result = self.assembler.assemble_page(regions, num_columns=2)
        assert "\\begin{multicols}{2}" in result
        assert "\\end{multicols}" in result

    def test_reading_order(self):
        """Test that regions are sorted by reading order."""
        regions = [
            {"type": RegionType.TEXT, "latex": "Bottom", "bbox": (0, 100, 100, 130)},
            {"type": RegionType.TEXT, "latex": "Top", "bbox": (0, 0, 100, 30)},
        ]
        result = self.assembler.assemble_page(regions)
        top_pos = result.find("Top")
        bottom_pos = result.find("Bottom")
        assert top_pos < bottom_pos

    def test_assemble_document(self):
        """Test full document assembly."""
        pages = [
            {
                "regions": [
                    {"type": RegionType.TEXT, "latex": "Page 1 content", "bbox": (0, 0, 100, 30)},
                ],
                "num_columns": 1,
            },
            {
                "regions": [
                    {"type": RegionType.TEXT, "latex": "Page 2 content", "bbox": (0, 0, 100, 30)},
                ],
                "num_columns": 1,
            },
        ]
        tm = TemplateManager()
        result = self.assembler.assemble_document(pages, template_manager=tm)
        assert "\\begin{document}" in result
        assert "Page 1 content" in result
        assert "Page 2 content" in result
        assert "\\newpage" in result
