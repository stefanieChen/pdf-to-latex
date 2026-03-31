"""LaTeX compilation and verification."""

import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger("pdf2latex.validation.compiler")


class LatexCompiler:
    """Compile LaTeX documents and verify output."""

    def __init__(self, compiler: str = "xelatex", timeout: int = 60):
        """Initialize LatexCompiler.

        Args:
            compiler: LaTeX compiler command (xelatex recommended for CJK).
            timeout: Compilation timeout in seconds.
        """
        self.compiler = compiler
        self.timeout = timeout

    def is_available(self) -> bool:
        """Check if the LaTeX compiler is installed and accessible.

        Returns:
            True if compiler is found on PATH.
        """
        return shutil.which(self.compiler) is not None

    def compile(
        self,
        tex_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Tuple[bool, str, Optional[Path]]:
        """Compile a .tex file to PDF.

        Args:
            tex_path: Path to .tex file.
            output_dir: Directory for output PDF. Defaults to tex_path's parent.

        Returns:
            Tuple of (success, log_output, pdf_path_or_None).
        """
        tex_path = Path(tex_path).resolve()
        if not tex_path.exists():
            return False, f"File not found: {tex_path}", None

        if output_dir is None:
            output_dir = tex_path.parent
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.compiler,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={output_dir}",
            str(tex_path),
        ]

        logger.info("Compiling: %s", " ".join(cmd))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(tex_path.parent),
            )

            log_output = result.stdout + result.stderr
            pdf_name = tex_path.stem + ".pdf"
            pdf_path = output_dir / pdf_name

            if result.returncode == 0 and pdf_path.exists():
                logger.info("Compilation successful: %s", pdf_path)
                return True, log_output, pdf_path
            else:
                errors = self._extract_errors(log_output)
                logger.warning("Compilation failed: %s", errors[:500])
                return False, log_output, None

        except subprocess.TimeoutExpired:
            logger.error("Compilation timed out after %ds", self.timeout)
            return False, f"Compilation timed out after {self.timeout}s", None
        except FileNotFoundError:
            logger.error("Compiler not found: %s", self.compiler)
            return False, f"Compiler not found: {self.compiler}", None

    def compile_string(
        self,
        latex_code: str,
        output_dir: Optional[Path] = None,
        filename: str = "output",
    ) -> Tuple[bool, str, Optional[Path]]:
        """Compile LaTeX code from a string.

        Args:
            latex_code: Complete LaTeX document string.
            output_dir: Directory for output files.
            filename: Base filename (without extension).

        Returns:
            Tuple of (success, log_output, pdf_path_or_None).
        """
        if output_dir is None:
            output_dir = Path(tempfile.mkdtemp(prefix="pdf2latex_"))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tex_path = output_dir / f"{filename}.tex"
        tex_path.write_text(latex_code, encoding="utf-8")

        return self.compile(tex_path, output_dir)

    def _extract_errors(self, log: str) -> str:
        """Extract error lines from LaTeX log output.

        Args:
            log: Full LaTeX log output.

        Returns:
            Extracted error lines.
        """
        error_lines = []
        for line in log.split("\n"):
            if line.startswith("!") or "Error" in line or "Fatal" in line:
                error_lines.append(line.strip())
        return "\n".join(error_lines) if error_lines else "Unknown compilation error"

    def cleanup_aux_files(self, tex_path: Path) -> None:
        """Remove auxiliary files generated during compilation.

        Args:
            tex_path: Path to the .tex file.
        """
        tex_path = Path(tex_path)
        aux_extensions = [".aux", ".log", ".out", ".toc", ".lof", ".lot", ".fls", ".fdb_latexmk", ".synctex.gz"]
        for ext in aux_extensions:
            aux_file = tex_path.with_suffix(ext)
            if aux_file.exists():
                aux_file.unlink()
