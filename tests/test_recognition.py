"""Unit tests for recognition modules."""

import numpy as np
import pytest

from src.recognition.text_recognizer import TextRecognizer, TextLine
from src.recognition.formula_recognizer import FormulaRecognizer
from src.recognition.table_recognizer import TableRecognizer


class TestTextRecognizer:
    """Tests for TextRecognizer (unit-level, no PaddleOCR required)."""

    def test_escape_latex(self):
        """Test LaTeX special character escaping."""
        assert TextRecognizer._escape_latex("a & b") == r"a \& b"
        assert TextRecognizer._escape_latex("100%") == r"100\%"
        assert TextRecognizer._escape_latex("$x$") == r"\$x\$"
        assert TextRecognizer._escape_latex("a_b") == r"a\_b"
        assert TextRecognizer._escape_latex("a#b") == r"a\#b"

    def test_group_paragraphs(self):
        """Test paragraph grouping based on vertical spacing."""
        recognizer = TextRecognizer.__new__(TextRecognizer)

        lines = [
            TextLine("Line 1", 0.9, (0, 0, 100, 20)),
            TextLine("Line 2", 0.9, (0, 22, 100, 42)),
            TextLine("Line 3", 0.9, (0, 80, 100, 100)),  # Large gap
        ]

        paragraphs = recognizer._group_paragraphs(lines)
        assert len(paragraphs) == 2
        assert len(paragraphs[0]) == 2
        assert len(paragraphs[1]) == 1


class TestFormulaRecognizer:
    """Tests for FormulaRecognizer (unit-level)."""

    def test_clean_latex_response_code_block(self):
        """Test cleaning LaTeX from code blocks."""
        response = "```latex\nx^2 + y^2 = z^2\n```"
        assert FormulaRecognizer._clean_latex_response(response) == "x^2 + y^2 = z^2"

    def test_clean_latex_response_dollar_signs(self):
        """Test cleaning LaTeX with dollar signs."""
        assert FormulaRecognizer._clean_latex_response("$x^2$") == "x^2"
        assert FormulaRecognizer._clean_latex_response("$$x^2$$") == "x^2"

    def test_clean_latex_response_display_math(self):
        """Test cleaning LaTeX with display math delimiters."""
        assert FormulaRecognizer._clean_latex_response("\\[x^2\\]") == "x^2"

    def test_clean_latex_response_plain(self):
        """Test that plain LaTeX passes through."""
        assert FormulaRecognizer._clean_latex_response("x^2 + y^2") == "x^2 + y^2"

    def test_format_inline(self):
        """Test inline math formatting."""
        recognizer = FormulaRecognizer()
        assert recognizer.format_as_inline("x^2") == "$x^2$"

    def test_format_display(self):
        """Test display math formatting."""
        recognizer = FormulaRecognizer()
        result = recognizer.format_as_display("x^2")
        assert "\\[" in result
        assert "x^2" in result
        assert "\\]" in result


class TestTableRecognizer:
    """Tests for TableRecognizer (unit-level)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recognizer = TableRecognizer.__new__(TableRecognizer)
        self.recognizer.use_gpu = False
        self.recognizer._engine = None

    def test_parse_html_simple(self):
        """Test parsing a simple HTML table."""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        rows = self.recognizer._parse_html_table(html)
        assert len(rows) == 2
        assert rows[0] == ["A", "B"]
        assert rows[1] == ["1", "2"]

    def test_parse_html_with_th(self):
        """Test parsing HTML table with th elements."""
        html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td>Alice</td><td>30</td></tr></table>"
        rows = self.recognizer._parse_html_table(html)
        assert len(rows) == 2
        assert rows[0] == ["Name", "Age"]

    def test_html_to_latex(self):
        """Test HTML to LaTeX conversion."""
        html = "<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>"
        latex = self.recognizer.html_to_latex(html)

        assert "\\begin{table}" in latex
        assert "\\begin{tabular}" in latex
        assert "\\end{tabular}" in latex
        assert "\\end{table}" in latex
        assert "A & B" in latex
        assert "1 & 2" in latex

    def test_escape_cell(self):
        """Test LaTeX escaping in table cells."""
        assert TableRecognizer._escape_cell("10%") == r"10\%"
        assert TableRecognizer._escape_cell("A & B") == r"A \& B"

    def test_empty_html(self):
        """Test handling of empty HTML."""
        rows = self.recognizer._parse_html_table("")
        assert rows == []
        latex = self.recognizer.html_to_latex("")
        assert latex == ""
