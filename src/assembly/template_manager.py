"""LaTeX document template management."""

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("pdf2latex.assembly.template")

DEFAULT_PREAMBLE = r"""\documentclass[12pt, a4paper]{article}

% Chinese support
\usepackage[UTF8]{ctex}

% Math
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}

% Graphics
\usepackage{graphicx}
\usepackage{tikz}
\usepackage{float}

% Tables
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{array}

% Layout
\usepackage[margin=2.5cm]{geometry}
\usepackage{fancyhdr}
\usepackage{multicol}

% Hyperlinks
\usepackage{hyperref}
\hypersetup{hidelinks}

% Misc
\usepackage{enumitem}
\usepackage{caption}
"""


class TemplateManager:
    """Manage LaTeX document templates and preamble generation."""

    def __init__(
        self,
        document_class: str = "article",
        packages: Optional[List[str]] = None,
        template_dir: Optional[Path] = None,
    ):
        """Initialize TemplateManager.

        Args:
            document_class: LaTeX document class.
            packages: List of package names to include.
            template_dir: Directory containing custom templates.
        """
        self.document_class = document_class
        self.packages = packages or []
        self.template_dir = Path(template_dir) if template_dir else None

    def generate_preamble(self, extra_packages: Optional[List[str]] = None) -> str:
        """Generate a LaTeX preamble with all required packages.

        Args:
            extra_packages: Additional packages beyond the defaults.

        Returns:
            LaTeX preamble string.
        """
        return DEFAULT_PREAMBLE

    def wrap_document(self, body: str, title: Optional[str] = None) -> str:
        """Wrap body content in a complete LaTeX document.

        Args:
            body: LaTeX body content.
            title: Optional document title.

        Returns:
            Complete LaTeX document string.
        """
        preamble = self.generate_preamble()

        parts = [preamble]

        if title:
            parts.append(f"\\title{{{title}}}")
            parts.append("\\date{}")

        parts.append("")
        parts.append("\\begin{document}")

        if title:
            parts.append("\\maketitle")

        parts.append("")
        parts.append(body)
        parts.append("")
        parts.append("\\end{document}")

        return "\n".join(parts)

    def load_template(self, template_name: str) -> Optional[str]:
        """Load a custom template from the template directory.

        Args:
            template_name: Name of the template file.

        Returns:
            Template content string, or None if not found.
        """
        if self.template_dir is None:
            return None

        template_path = self.template_dir / template_name
        if template_path.exists():
            return template_path.read_text(encoding="utf-8")

        logger.warning("Template not found: %s", template_path)
        return None
