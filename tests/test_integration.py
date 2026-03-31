"""Integration tests for the PDF-to-LaTeX system.

These tests verify end-to-end workflows using synthetic test data
that does not require PaddleOCR, Ollama, or DeTikZify to be running.
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config import load_config, PROJECT_ROOT
from src.preprocessing.pdf_converter import PdfConverter
from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.batch_loader import BatchLoader
from src.layout.layout_detector import BBox, DetectedRegion, RegionType
from src.recognition.text_recognizer import TextRecognizer
from src.recognition.formula_recognizer import FormulaRecognizer
from src.recognition.table_recognizer import TableRecognizer
from src.assembly.layout_assembler import LayoutAssembler
from src.assembly.template_manager import TemplateManager
from src.validation.latex_compiler import LatexCompiler
from src.validation.visual_comparator import VisualComparator


class TestConfigIntegration:
    """Test config loading with real settings.yaml."""

    def test_load_default_config(self):
        """Test loading the default settings.yaml."""
        config = load_config()
        assert "ollama" in config
        assert "preprocessing" in config
        assert "layout" in config
        assert "recognition" in config
        assert "assembly" in config
        assert "validation" in config
        assert "server" in config
        assert "paths" in config

    def test_paths_are_absolute(self):
        """Test that paths in config are resolved to absolute paths."""
        config = load_config()
        for key, val in config.get("paths", {}).items():
            assert Path(val).is_absolute(), f"Path '{key}' is not absolute: {val}"

    def test_directories_created(self):
        """Test that required directories are created by config loader."""
        config = load_config()
        for key, val in config.get("paths", {}).items():
            assert Path(val).exists(), f"Directory '{key}' was not created: {val}"


class TestPreprocessingIntegration:
    """Test preprocessing pipeline end-to-end."""

    def test_enhance_then_load(self):
        """Test enhancing an image and loading it back."""
        # Create a synthetic document image with text-like features
        img = np.ones((800, 600, 3), dtype=np.uint8) * 240
        # Add some dark horizontal lines (simulating text)
        for y in range(100, 700, 40):
            cv2.line(img, (50, y), (550, y), (30, 30, 30), 2)

        enhancer = ImageEnhancer()
        enhanced = enhancer.enhance(img, denoise=True, deskew=True)

        assert enhanced is not None
        assert enhanced.shape[0] > 0 and enhanced.shape[1] > 0

        # Save and reload
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            cv2.imwrite(f.name, enhanced)
            tmp_path = Path(f.name)

        try:
            loader = BatchLoader()
            loaded = loader.load_image(tmp_path)
            assert loaded.shape == enhanced.shape
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_batch_classify_and_load(self):
        """Test batch loading and classification."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # Create test files
            img = np.ones((100, 100, 3), dtype=np.uint8) * 128
            cv2.imwrite(str(tmp / "page1.png"), img)
            cv2.imwrite(str(tmp / "page2.jpg"), img)
            (tmp / "notes.txt").write_text("not an image")

            loader = BatchLoader()
            files = loader.scan_directory(tmp)
            pdfs, images = loader.classify_files(files)

            assert len(pdfs) == 0
            assert len(images) == 2

            loaded = loader.load_images(images)
            assert len(loaded) == 2


class TestAssemblyIntegration:
    """Test full document assembly pipeline."""

    def test_full_document_assembly(self):
        """Test assembling a multi-page document with mixed content types."""
        assembler = LayoutAssembler()
        template_mgr = TemplateManager()

        pages = [
            {
                "regions": [
                    {"type": RegionType.TITLE, "latex": "Introduction", "bbox": (50, 10, 500, 50)},
                    {"type": RegionType.TEXT, "latex": "This is the first paragraph of the document.", "bbox": (50, 60, 500, 120)},
                    {"type": RegionType.FORMULA, "latex": "E = mc^2", "bbox": (100, 130, 400, 180)},
                    {"type": RegionType.TEXT, "latex": "The above formula shows mass-energy equivalence.", "bbox": (50, 190, 500, 240)},
                ],
                "num_columns": 1,
            },
            {
                "regions": [
                    {"type": RegionType.TITLE, "latex": "Data Analysis", "bbox": (50, 10, 500, 50)},
                    {
                        "type": RegionType.TABLE,
                        "latex": (
                            "\\begin{table}[H]\n\\centering\n"
                            "\\begin{tabular}{|c|c|}\n\\hline\n"
                            "Name & Value \\\\\n\\hline\n"
                            "Alpha & 1.5 \\\\\n\\hline\n"
                            "Beta & 2.3 \\\\\n\\hline\n"
                            "\\end{tabular}\n\\end{table}"
                        ),
                        "bbox": (50, 60, 500, 250),
                    },
                    {
                        "type": RegionType.FIGURE,
                        "latex": (
                            "\\begin{figure}[H]\n\\centering\n"
                            "\\begin{tikzpicture}\n"
                            "\\draw (0,0) circle (1);\n"
                            "\\end{tikzpicture}\n"
                            "\\end{figure}"
                        ),
                        "bbox": (50, 260, 500, 450),
                    },
                ],
                "num_columns": 1,
            },
        ]

        doc = assembler.assemble_document(pages, template_manager=template_mgr, title="Test Document")

        # Verify document structure
        assert "\\documentclass" in doc
        assert "\\begin{document}" in doc
        assert "\\end{document}" in doc
        assert "\\maketitle" in doc

        # Verify content
        assert "\\section*{Introduction}" in doc
        assert "This is the first paragraph" in doc
        assert "E = mc^2" in doc
        assert "\\section*{Data Analysis}" in doc
        assert "\\begin{tabular}" in doc
        assert "\\begin{tikzpicture}" in doc
        assert "\\newpage" in doc

    def test_multicol_assembly(self):
        """Test two-column assembly."""
        assembler = LayoutAssembler()

        regions = [
            {"type": RegionType.TEXT, "latex": "Left column text.", "bbox": (10, 10, 250, 50)},
            {"type": RegionType.TEXT, "latex": "Right column text.", "bbox": (260, 10, 500, 50)},
        ]

        result = assembler.assemble_page(regions, num_columns=2)
        assert "\\begin{multicols}{2}" in result
        assert "\\end{multicols}" in result


class TestVisualComparatorIntegration:
    """Test visual comparison with realistic synthetic images."""

    def test_document_like_comparison(self):
        """Test SSIM with document-like images."""
        # Create a synthetic document page
        doc = np.ones((800, 600), dtype=np.uint8) * 255
        # Add text-like horizontal stripes
        for y in range(100, 700, 30):
            cv2.rectangle(doc, (50, y), (550, y + 8), 0, -1)

        # Create a slightly degraded version (simulating scan)
        degraded = doc.copy()
        noise = np.random.normal(0, 5, degraded.shape).astype(np.int16)
        degraded = np.clip(degraded.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        comparator = VisualComparator(ssim_threshold=0.85)
        score, passes = comparator.compare(doc, degraded)

        assert score > 0.8
        assert isinstance(passes, bool)

    def test_mismatched_documents(self):
        """Test SSIM catches significant layout differences."""
        doc1 = np.ones((800, 600), dtype=np.uint8) * 255
        for y in range(100, 400, 30):
            cv2.rectangle(doc1, (50, y), (550, y + 8), 0, -1)

        doc2 = np.ones((800, 600), dtype=np.uint8) * 255
        for y in range(400, 700, 30):
            cv2.rectangle(doc2, (50, y), (550, y + 8), 0, -1)

        comparator = VisualComparator(ssim_threshold=0.85)
        score, passes = comparator.compare(doc1, doc2)
        assert score < 0.85


class TestCompilerIntegration:
    """Test LaTeX compiler integration."""

    def test_compiler_availability_check(self):
        """Test that compiler availability check works."""
        compiler = LatexCompiler()
        # Just check the method runs without error
        result = compiler.is_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(
        not LatexCompiler().is_available(),
        reason="LaTeX compiler not installed",
    )
    def test_compile_simple_document(self):
        """Test compiling a simple LaTeX document."""
        compiler = LatexCompiler(compiler="xelatex")

        latex = (
            "\\documentclass{article}\n"
            "\\begin{document}\n"
            "Hello, World!\n"
            "\\end{document}\n"
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            success, log, pdf_path = compiler.compile_string(
                latex, output_dir=Path(tmp_dir), filename="test"
            )
            assert success is True
            assert pdf_path is not None
            assert pdf_path.exists()


class TestServerIntegration:
    """Test FastAPI server endpoints (without starting the server)."""

    def test_app_creates(self):
        """Test that the FastAPI app object can be created."""
        from server import app
        assert app is not None
        assert app.title == "PDF-to-LaTeX Conversion API"

    def test_static_dir_exists(self):
        """Test that static directory exists for frontend."""
        from server import STATIC_DIR
        assert STATIC_DIR.exists()
        assert (STATIC_DIR / "index.html").exists()
        assert (STATIC_DIR / "css" / "style.css").exists()
        assert (STATIC_DIR / "js" / "app.js").exists()
