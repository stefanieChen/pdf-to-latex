"""LLM-based review of generated LaTeX against source documents."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("pdf2latex.validation.reviewer")


class LLMReviewer:
    """Use VLM to review generated LaTeX by comparing source and output images."""

    def __init__(
        self,
        ollama_client=None,
        vlm_model: str = "minicpm-v",
        llm_model: str = "qwen2.5:7b",
        max_fix_rounds: int = 3,
    ):
        """Initialize LLMReviewer.

        Args:
            ollama_client: OllamaClient instance.
            vlm_model: Vision-language model for visual comparison.
            llm_model: Text LLM for LaTeX code fixing.
            max_fix_rounds: Maximum auto-fix attempts.
        """
        self.ollama = ollama_client
        self.vlm_model = vlm_model
        self.llm_model = llm_model
        self.max_fix_rounds = max_fix_rounds

    def review(
        self,
        source_image: np.ndarray,
        generated_image: np.ndarray,
    ) -> dict:
        """Review generated output against source using VLM.

        Args:
            source_image: Original document image (BGR).
            generated_image: Generated PDF rendered as image (BGR).

        Returns:
            Dict with 'passed', 'differences', 'suggestions'.
        """
        if self.ollama is None:
            logger.warning("No Ollama client available for review")
            return {"passed": True, "differences": "", "suggestions": ""}

        # Save images to temp files
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as src_tmp:
            cv2.imwrite(src_tmp.name, source_image)
            src_path = src_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as gen_tmp:
            cv2.imwrite(gen_tmp.name, generated_image)
            gen_path = gen_tmp.name

        prompt = (
            "Compare these two document images. The first is the original source, "
            "the second is a LaTeX-generated reproduction. "
            "List any differences in: text content, layout, formulas, tables, or figures. "
            "If they match well, say 'PASS'. Otherwise describe each difference briefly."
        )

        try:
            response = self.ollama.generate(
                model=self.vlm_model,
                prompt=prompt,
                images=[src_path, gen_path],
                temperature=0.1,
            )

            passed = "PASS" in response.upper() and "FAIL" not in response.upper()

            result = {
                "passed": passed,
                "differences": response,
                "suggestions": "",
            }

            logger.info("LLM review: passed=%s", passed)
            return result

        except Exception as e:
            logger.warning("LLM review failed: %s", e)
            return {"passed": True, "differences": f"Review error: {e}", "suggestions": ""}
        finally:
            Path(src_path).unlink(missing_ok=True)
            Path(gen_path).unlink(missing_ok=True)

    def fix_latex(self, latex_code: str, error_log: str) -> Optional[str]:
        """Attempt to fix LaTeX compilation errors using LLM.

        Args:
            latex_code: Current LaTeX code that failed to compile.
            error_log: Compilation error log.

        Returns:
            Fixed LaTeX code, or None if fix fails.
        """
        if self.ollama is None:
            return None

        prompt = (
            "The following LaTeX code fails to compile. "
            "Fix the compilation errors and return the complete corrected LaTeX code.\n\n"
            f"Error log:\n```\n{error_log[:2000]}\n```\n\n"
            f"LaTeX code:\n```latex\n{latex_code[:8000]}\n```\n\n"
            "Return ONLY the complete fixed LaTeX code, no explanation."
        )

        try:
            response = self.ollama.generate(
                model=self.llm_model,
                prompt=prompt,
                temperature=0.1,
            )

            fixed = self._extract_latex_code(response)
            if fixed and "\\begin{document}" in fixed:
                logger.info("LLM fix generated: %d chars", len(fixed))
                return fixed

        except Exception as e:
            logger.warning("LLM fix failed: %s", e)

        return None

    def auto_fix_loop(
        self,
        latex_code: str,
        compiler,
        output_dir: Path,
    ) -> tuple:
        """Run an auto-fix loop: compile → fix → recompile up to max_fix_rounds.

        Args:
            latex_code: Initial LaTeX code.
            compiler: LatexCompiler instance.
            output_dir: Directory for output files.

        Returns:
            Tuple of (final_latex_code, success, pdf_path).
        """
        current_code = latex_code

        for attempt in range(self.max_fix_rounds):
            success, log_output, pdf_path = compiler.compile_string(
                current_code, output_dir, f"output_v{attempt}"
            )

            if success:
                logger.info("Compilation succeeded on attempt %d", attempt + 1)
                return current_code, True, pdf_path

            logger.info("Fix attempt %d/%d", attempt + 1, self.max_fix_rounds)
            fixed = self.fix_latex(current_code, log_output)
            if fixed is None:
                logger.warning("LLM could not fix the code")
                break
            current_code = fixed

        return current_code, False, None

    @staticmethod
    def _extract_latex_code(response: str) -> Optional[str]:
        """Extract LaTeX code from LLM response.

        Args:
            response: Raw LLM response text.

        Returns:
            Extracted LaTeX code or None.
        """
        text = response.strip()

        # Try to extract from code block
        if "```latex" in text:
            start = text.index("```latex") + 8
            end = text.index("```", start) if "```" in text[start:] else len(text)
            return text[start:end].strip()

        if "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start) if "```" in text[start:] else len(text)
            extracted = text[start:end].strip()
            if "\\begin{document}" in extracted:
                return extracted

        # If response looks like raw LaTeX
        if "\\documentclass" in text or "\\begin{document}" in text:
            return text

        return None
