"""LaTeX compilation and verification."""

import logging
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

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
            True if compiler is found on PATH or in common locations.
        """
        # First try shutil.which (standard method)
        if shutil.which(self.compiler):
            return True
        
        # On Windows, check common MiKTeX installation paths
        if platform.system() == "Windows" and self.compiler == "xelatex":
            common_paths = [
                r"C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe",
                r"C:\Program Files (x86)\MiKTeX\miktex\bin\xelatex.exe", 
                r"C:\miktex\miktex\bin\x64\xelatex.exe",
                r"C:\miktex\miktex\bin\xelatex.exe",
            ]
            for path in common_paths:
                if Path(path).exists():
                    return True
        
        # Fallback: try running the compiler directly
        try:
            result = subprocess.run(
                [self.compiler, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _get_compiler_command(self) -> Optional[str]:
        """Get the full command to run the compiler.
        
        Returns:
            Full path to compiler executable, or None if not found.
        """
        # First try shutil.which (standard method)
        compiler_path = shutil.which(self.compiler)
        if compiler_path:
            return compiler_path
        
        # On Windows, check common MiKTeX installation paths
        if platform.system() == "Windows" and self.compiler == "xelatex":
            common_paths = [
                r"C:\Program Files\MiKTeX\miktex\bin\x64\xelatex.exe",
                r"C:\Program Files (x86)\MiKTeX\miktex\bin\xelatex.exe", 
                r"C:\miktex\miktex\bin\x64\xelatex.exe",
                r"C:\miktex\miktex\bin\xelatex.exe",
            ]
            for path in common_paths:
                if Path(path).exists():
                    return path
        
        # Fallback: try the compiler name directly (might be in PATH)
        return self.compiler

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

        # Get the full compiler path
        compiler_cmd = self._get_compiler_command()
        if not compiler_cmd:
            return False, "Compiler not available", None

        cmd = [
            compiler_cmd,
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
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                cwd=str(tex_path.parent),
            )

            log_output = (result.stdout or "") + (result.stderr or "")
            pdf_name = tex_path.stem + ".pdf"
            pdf_path = output_dir / pdf_name

            if result.returncode == 0 and pdf_path.exists():
                logger.info("Compilation successful: %s", pdf_path)
                return True, log_output, pdf_path
            else:
                errors = self._extract_errors(log_output)
                logger.warning("Compilation failed: %s", errors[:500])
                
                # Try to fix common errors and retry once
                if any(error in log_output for error in ["Missing $ inserted", "Extra $", "Undefined control sequence"]):
                    logger.info("Attempting to fix common LaTeX errors...")
                    
                    # Read the original tex file
                    try:
                        original_latex = tex_path.read_text(encoding="utf-8")
                        fixed_latex = self.fix_common_latex_errors(original_latex)
                        
                        # Write fixed version and retry
                        tex_path.write_text(fixed_latex, encoding="utf-8")
                        
                        # Retry compilation
                        result_retry = subprocess.run(
                            cmd,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                            timeout=self.timeout,
                            cwd=str(tex_path.parent),
                        )
                        
                        log_output_retry = (result_retry.stdout or "") + (result_retry.stderr or "")
                        
                        if result_retry.returncode == 0 and pdf_path.exists():
                            logger.info("Compilation successful after fixing: %s", pdf_path)
                            return True, log_output_retry, pdf_path
                        else:
                            # Restore original and return failure
                            tex_path.write_text(original_latex, encoding="utf-8")
                            errors_retry = self._extract_errors(log_output_retry)
                            logger.warning("Compilation failed even after fixing: %s", errors_retry[:500])
                            return False, f"Original error: {errors}\n\nAfter fixing attempt: {errors_retry}", None
                            
                    except Exception as e:
                        logger.error("Error during automatic fixing: %s", e)
                
                return False, errors, None

        except subprocess.TimeoutExpired:
            logger.error("Compilation timed out after %ds", self.timeout)
            return False, f"Compilation timed out after {self.timeout}s", None
        except FileNotFoundError:
            logger.error("Compiler not found: %s", compiler_cmd)
            return False, f"Compiler not found: {compiler_cmd}", None

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
            Extracted error lines with context.
        """
        error_lines = []
        lines = log.split("\n")
        
        for i, line in enumerate(lines):
            # Look for error indicators
            if (line.startswith("!") or 
                "Error" in line or 
                "Fatal" in line or
                "Missing" in line or
                "Undefined" in line or
                "Illegal" in line):
                
                # Include the error line and some context
                error_lines.append(line.strip())
                
                # Add next few lines for context (often contain the problematic LaTeX)
                for j in range(1, min(4, len(lines) - i)):
                    next_line = lines[i + j].strip()
                    if next_line and not next_line.startswith("l.") and not next_line.startswith("..."):
                        error_lines.append(f"  {next_line}")
                    elif next_line.startswith("l."):
                        error_lines.append(f"  {next_line}")
                        break
        
        # If no specific errors found, return a generic message
        if not error_lines:
            # Look for common LaTeX error patterns
            common_errors = [
                "Missing $ inserted",
                "Extra $", 
                "Undefined control sequence",
                "Missing { inserted",
                "Missing } inserted",
                "Missing \\begin inserted",
                "Missing \\end inserted"
            ]
            
            for error in common_errors:
                if error in log:
                    return f"LaTeX Error: {error}\nCheck the generated LaTeX code for syntax issues."
            
            return "Unknown compilation error - check LaTeX syntax and packages"
        
        return "\n".join(error_lines)

    def pre_validate(self, latex_code: str) -> List[str]:
        """Quick syntax check without spawning xelatex.

        Catches trivially broken LaTeX (unmatched delimiters, missing
        document structure) so the LLM fix loop can be invoked directly
        without wasting a 60 s compilation attempt.

        Args:
            latex_code: Complete LaTeX document string.

        Returns:
            List of error description strings.  Empty list means no
            obvious problems detected.
        """
        errors: List[str] = []

        # Check document structure
        if "\\begin{document}" not in latex_code:
            errors.append("Missing \\begin{document}")
        if "\\end{document}" not in latex_code:
            errors.append("Missing \\end{document}")

        # Check matched $ delimiters (ignoring \$ escapes)
        stripped = re.sub(r"\\\$", "", latex_code)  # remove escaped $
        # Also ignore $$ (display math) by replacing with placeholder
        stripped = stripped.replace("$$", "DD")
        dollar_count = stripped.count("$")
        if dollar_count % 2 != 0:
            errors.append(f"Unmatched $ delimiter (found {dollar_count} unescaped $)")

        # Check matched braces (outside comments)
        depth = 0
        for line in latex_code.split("\n"):
            # Strip comments
            uncommented = re.sub(r"(?<!\\)%.*", "", line)
            for ch in uncommented:
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                if depth < 0:
                    errors.append("Extra closing brace }")
                    depth = 0
        if depth > 0:
            errors.append(f"Unclosed braces: {depth} opening {{ without matching }}")

        # Check matched \begin / \end environments
        begins = re.findall(r"\\begin\{([^}]+)\}", latex_code)
        ends = re.findall(r"\\end\{([^}]+)\}", latex_code)
        begin_counts: dict = {}
        end_counts: dict = {}
        for env in begins:
            begin_counts[env] = begin_counts.get(env, 0) + 1
        for env in ends:
            end_counts[env] = end_counts.get(env, 0) + 1
        for env in set(list(begin_counts.keys()) + list(end_counts.keys())):
            b = begin_counts.get(env, 0)
            e = end_counts.get(env, 0)
            if b != e:
                errors.append(f"Mismatched environment '{env}': {b} \\begin vs {e} \\end")

        if errors:
            logger.info("Pre-validation found %d issue(s): %s", len(errors), "; ".join(errors))
        return errors

    def fix_common_latex_errors(self, latex_code: str) -> str:
        """Attempt to fix common LaTeX syntax errors (math-mode aware).

        Unlike a naive regex approach, this method tracks whether we are
        inside a math environment (``$...$``, ``\\[...\\]``,
        ``\\begin{equation}`` etc.) and only escapes special characters
        in *text* regions, leaving math content untouched.

        Args:
            latex_code: Original LaTeX code.

        Returns:
            Fixed LaTeX code.
        """
        lines = latex_code.split("\n")
        fixed_lines: List[str] = []
        in_math_env = False

        # Environments that are math-mode
        math_envs = {
            "equation", "equation*", "align", "align*", "gather",
            "gather*", "multline", "multline*", "eqnarray", "eqnarray*",
            "math", "displaymath", "tikzpicture",
        }

        for line in lines:
            # Track math environment boundaries
            for env in math_envs:
                if f"\\begin{{{env}}}" in line:
                    in_math_env = True
                if f"\\end{{{env}}}" in line:
                    in_math_env = False

            if in_math_env:
                fixed_lines.append(line)
                continue

            # For text-mode lines, fix character by character
            fixed_lines.append(self._fix_text_line(line))

        return "\n".join(fixed_lines)

    @staticmethod
    def _fix_text_line(line: str) -> str:
        """Fix special characters in a single text-mode line.

        Tracks inline math (``$...$``) to avoid escaping inside math spans.

        Args:
            line: A single line of LaTeX source.

        Returns:
            Line with special characters properly escaped in text regions.
        """
        # Strip comment portion
        comment_match = re.search(r"(?<!\\)%", line)
        if comment_match:
            text_part = line[:comment_match.start()]
            comment_part = line[comment_match.start():]
        else:
            text_part = line
            comment_part = ""

        # Split by inline math delimiters, preserving them
        # Odd-indexed segments are inside $...$
        parts = re.split(r"(?<!\\)(\$)", text_part)

        result: List[str] = []
        in_inline_math = False
        for part in parts:
            if part == "$":
                in_inline_math = not in_inline_math
                result.append(part)
            elif in_inline_math:
                result.append(part)
            else:
                # Escape only in text mode, skip already-escaped chars
                fixed = part
                fixed = re.sub(r"(?<!\\)_(?![{a-zA-Z])", r"\\_", fixed)
                fixed = re.sub(r"(?<!\\)\^(?![{a-zA-Z])", r"\\^{}", fixed)
                result.append(fixed)

        return "".join(result) + comment_part

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
