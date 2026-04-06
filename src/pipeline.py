"""Main pipeline orchestrating the full PDF/Image to LaTeX conversion."""

import concurrent.futures
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import load_config, setup_logging, PROJECT_ROOT
from src.ollama_client import OllamaClient
from src.detikzify_client import DeTikZifyClient
from src.model_scheduler import ModelScheduler, ModelType
from src.preprocessing.pdf_converter import PdfConverter
from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.batch_loader import BatchLoader
from src.layout.layout_detector import LayoutDetector, RegionType
from src.recognition.text_recognizer import TextRecognizer
from src.recognition.formula_recognizer import FormulaRecognizer
from src.recognition.table_recognizer import TableRecognizer
from src.recognition.figure_recognizer import FigureRecognizer
from src.assembly.layout_assembler import LayoutAssembler
from src.assembly.template_manager import TemplateManager
from src.validation.latex_compiler import LatexCompiler
from src.validation.visual_comparator import VisualComparator
from src.validation.llm_reviewer import LLMReviewer
from src.monitoring.mlflow_tracker import init_mlflow, log_conversion_run

logger = logging.getLogger("pdf2latex.pipeline")


class PipelineStage(Enum):
    """Stages of the conversion pipeline."""
    INIT = "init"
    PREPROCESSING = "preprocessing"
    LAYOUT = "layout_analysis"
    RECOGNITION = "recognition"
    ASSEMBLY = "assembly"
    VALIDATION = "validation"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TaskStatus:
    """Status of a conversion task."""
    task_id: str
    stage: PipelineStage = PipelineStage.INIT
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None
    output_tex: Optional[Path] = None
    output_pdf: Optional[Path] = None
    stage_times: Dict[str, float] = field(default_factory=dict)
    ssim_score: Optional[float] = None


class Pipeline:
    """Main conversion pipeline: PDF/Image → LaTeX."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the pipeline with all components.

        Args:
            config: Configuration dictionary. Loads from settings.yaml if None.
        """
        self.config = config or load_config()
        self._init_components()
        self._progress_callback: Optional[Callable] = None

        # Initialize MLflow tracking if enabled
        init_mlflow(self.config)

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        cfg = self.config

        # Preprocessing
        pre_cfg = cfg.get("preprocessing", {})
        self.pdf_converter = PdfConverter(dpi=pre_cfg.get("dpi", 300))
        self.image_enhancer = ImageEnhancer(
            denoise_strength=pre_cfg.get("denoise_strength", 10),
            binary_block_size=pre_cfg.get("binary_block_size", 11),
            binary_constant=pre_cfg.get("binary_constant", 2),
        )
        self.batch_loader = BatchLoader(
            supported_formats=pre_cfg.get("supported_formats"),
        )

        # Layout
        layout_cfg = cfg.get("layout", {})
        self.layout_detector = LayoutDetector(
            confidence_threshold=layout_cfg.get("confidence_threshold", 0.5),
            use_gpu=cfg.get("recognition", {}).get("ocr", {}).get("use_gpu", True),
        )

        # Clients
        ollama_cfg = cfg.get("ollama", {})
        self.ollama_client = OllamaClient(
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            timeout=ollama_cfg.get("timeout", 300),
        )

        dtk_cfg = cfg.get("detikzify", {})
        self.detikzify_client = DeTikZifyClient(
            model_name=dtk_cfg.get("model_name", "nllg/detikzify-v2.5-8b"),
            torch_dtype=dtk_cfg.get("torch_dtype", "bfloat16"),
            quantize_4bit=dtk_cfg.get("quantize_4bit", True),
            device_map=dtk_cfg.get("device_map", "auto"),
        )

        # Model scheduler
        self.scheduler = ModelScheduler(
            ollama_client=self.ollama_client,
            detikzify_client=self.detikzify_client,
            config=cfg,
        )

        # Recognition
        rec_cfg = cfg.get("recognition", {})
        ocr_cfg = rec_cfg.get("ocr", {})
        self.text_recognizer = TextRecognizer(
            lang=ocr_cfg.get("lang", "ch"),
            use_angle_cls=ocr_cfg.get("use_angle_cls", True),
            use_gpu=ocr_cfg.get("use_gpu", True),
        )
        self.formula_recognizer = FormulaRecognizer(
            ollama_client=self.ollama_client,
            ollama_model=ollama_cfg.get("models", {}).get("vlm", "minicpm-v"),
        )
        self.table_recognizer = TableRecognizer(
            use_gpu=ocr_cfg.get("use_gpu", True),
        )
        self.figure_recognizer = FigureRecognizer(
            detikzify_client=self.detikzify_client,
            ollama_client=self.ollama_client,
            model_scheduler=self.scheduler,
            ollama_vlm_model=ollama_cfg.get("models", {}).get("vlm", "minicpm-v"),
            mcts_timeout=dtk_cfg.get("mcts_timeout", 300),
        )

        # Assembly
        asm_cfg = cfg.get("assembly", {})
        self.assembler = LayoutAssembler()
        self.template_manager = TemplateManager(
            document_class=asm_cfg.get("document_class", "article"),
            packages=asm_cfg.get("packages", []),
            template_dir=PROJECT_ROOT / "config" / "latex_templates",
        )

        # Validation
        val_cfg = cfg.get("validation", {})
        self.compiler = LatexCompiler(
            compiler=val_cfg.get("compiler", "xelatex"),
            timeout=val_cfg.get("compile_timeout", 60),
        )
        self.comparator = VisualComparator(
            ssim_threshold=val_cfg.get("ssim_threshold", 0.85),
        )
        self.reviewer = LLMReviewer(
            ollama_client=self.ollama_client,
            vlm_model=ollama_cfg.get("models", {}).get("vlm", "minicpm-v"),
            llm_model=ollama_cfg.get("models", {}).get("llm", "qwen2.5:7b"),
            max_fix_rounds=val_cfg.get("max_fix_rounds", 3),
        )

        # Eagerly warm up PaddleOCR engines to avoid first-request cold start
        self._warm_up_engines()

    def set_progress_callback(self, callback: Callable[[TaskStatus], None]) -> None:
        """Set a callback function for progress updates.

        Args:
            callback: Function that takes a TaskStatus and handles progress reporting.
        """
        self._progress_callback = callback

    def _report_progress(self, status: TaskStatus) -> None:
        """Report progress through the callback if set."""
        if self._progress_callback:
            self._progress_callback(status)

    def _warm_up_engines(self) -> None:
        """Eagerly initialize PaddleOCR engines to avoid first-request cold start.

        Layout detection, text recognition, and table recognition all lazy-init
        their PaddleOCR backends on first call.  By calling ``_init_engine()``
        here we pay the cost once at startup rather than on the first conversion.
        """
        engines = [
            ("LayoutDetector", self.layout_detector),
            ("TextRecognizer", self.text_recognizer),
            ("TableRecognizer", self.table_recognizer),
        ]
        for name, engine in engines:
            try:
                if hasattr(engine, "_init_engine"):
                    engine._init_engine()
                    logger.info("Warm-up: %s engine ready", name)
            except Exception as e:
                logger.warning("Warm-up: %s init failed (will retry on first use): %s", name, e)

    def convert(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        task_id: Optional[str] = None,
        use_mcts: bool = False,
    ) -> TaskStatus:
        """Run the full conversion pipeline on a file or directory.

        Args:
            input_path: Path to PDF file, image file, or directory of images.
            output_dir: Output directory for .tex and .pdf files.
            task_id: Unique task identifier (auto-generated if None).
            use_mcts: Whether to use MCTS for figure recognition (slower, better).

        Returns:
            TaskStatus with results.
        """
        input_path = Path(input_path)
        if task_id is None:
            task_id = self._generate_task_id(input_path)

        if output_dir is None:
            output_dir = Path(self.config.get("paths", {}).get("output_dir", "data/output"))
        output_dir = Path(output_dir) / task_id
        output_dir.mkdir(parents=True, exist_ok=True)

        status = TaskStatus(task_id=task_id)
        start_time = time.time()

        try:
            # Stage 1: Preprocessing
            stage_t0 = time.perf_counter()
            status.stage = PipelineStage.PREPROCESSING
            status.progress = 0.1
            status.message = "Preprocessing input files..."
            self._report_progress(status)

            page_images = self._preprocess(input_path, output_dir)
            logger.info("Preprocessed %d pages", len(page_images))
            status.stage_times["preprocessing"] = round(time.perf_counter() - stage_t0, 2)

            # Stage 2: Layout Analysis
            stage_t0 = time.perf_counter()
            status.stage = PipelineStage.LAYOUT
            status.progress = 0.25
            status.message = "Analyzing document layout..."
            self._report_progress(status)

            page_layouts = []
            for i, (img_path, img) in enumerate(page_images):
                regions = self.layout_detector.detect(img, page_num=i)
                num_cols = self.layout_detector.detect_columns(regions, img.shape[1])
                page_layouts.append({
                    "image": img,
                    "image_path": img_path,
                    "regions": regions,
                    "num_columns": num_cols,
                    "page_num": i,
                })
            status.stage_times["layout_analysis"] = round(time.perf_counter() - stage_t0, 2)

            # Stage 3: Recognition
            stage_t0 = time.perf_counter()
            status.stage = PipelineStage.RECOGNITION
            status.progress = 0.4
            status.message = "Recognizing content..."
            self._report_progress(status)

            page_results = []
            for i, page in enumerate(page_layouts):
                recognized = self._recognize_page(page, output_dir, use_mcts)
                page_results.append({
                    "regions": recognized,
                    "num_columns": page["num_columns"],
                })
                status.progress = 0.4 + (0.3 * (i + 1) / len(page_layouts))
                status.message = f"Recognizing page {i + 1}/{len(page_layouts)}..."
                self._report_progress(status)
            status.stage_times["recognition"] = round(time.perf_counter() - stage_t0, 2)

            # Stage 4: Assembly
            stage_t0 = time.perf_counter()
            status.stage = PipelineStage.ASSEMBLY
            status.progress = 0.75
            status.message = "Assembling LaTeX document..."
            self._report_progress(status)

            latex_code = self.assembler.assemble_document(
                pages=page_results,
                template_manager=self.template_manager,
                title=input_path.stem,
            )

            tex_path = output_dir / "output.tex"
            tex_path.write_text(latex_code, encoding="utf-8")
            status.output_tex = tex_path
            logger.info("LaTeX document written: %s (%d chars)", tex_path, len(latex_code))
            status.stage_times["assembly"] = round(time.perf_counter() - stage_t0, 2)

            # Stage 5: Validation
            stage_t0 = time.perf_counter()
            status.stage = PipelineStage.VALIDATION
            status.progress = 0.85
            status.message = "Compiling and validating..."
            self._report_progress(status)

            if self.compiler.is_available():
                # Pre-validate: catch trivially broken LaTeX without spawning xelatex
                pre_errors = self.compiler.pre_validate(latex_code)
                if pre_errors:
                    logger.info("Pre-validation issues: %s — sending to LLM fix", pre_errors)

                latex_code, success, pdf_path = self.reviewer.auto_fix_loop(
                    latex_code, self.compiler, output_dir
                )

                if success and pdf_path:
                    status.output_pdf = pdf_path
                    # Update tex with fixed version
                    tex_path.write_text(latex_code, encoding="utf-8")

                    # Compute SSIM between source and output (Phase 5c)
                    try:
                        if page_images:
                            source_img = page_images[0][1]  # first page BGR
                            rendered_img = self.pdf_converter.convert_page_to_numpy(
                                pdf_path, page_num=0,
                            )
                            ssim_score, _ = self.comparator.compare(source_img, rendered_img)
                            status.ssim_score = round(ssim_score, 4)
                            logger.info("Output SSIM score: %.4f", ssim_score)
                    except Exception as ssim_err:
                        logger.debug("SSIM comparison skipped: %s", ssim_err)
                else:
                    logger.warning("Compilation failed after auto-fix attempts")
            else:
                logger.warning("LaTeX compiler not available, skipping compilation")
            status.stage_times["validation"] = round(time.perf_counter() - stage_t0, 2)

            # Done
            status.stage = PipelineStage.COMPLETE
            status.progress = 1.0
            elapsed = time.time() - start_time
            status.message = f"Conversion complete in {elapsed:.1f}s"
            self._report_progress(status)

            # Release models
            self.scheduler.release()

            # Log to MLflow (include per-stage timings and SSIM)
            metrics = {
                "total_time_s": round(elapsed, 2),
                "num_pages": len(page_images),
                "latex_length": len(latex_code),
                "compilation_success": 1.0 if status.output_pdf else 0.0,
            }
            metrics.update({f"stage_{k}_s": v for k, v in status.stage_times.items()})
            if status.ssim_score is not None:
                metrics["ssim_score"] = status.ssim_score

            log_conversion_run(
                task_id=task_id,
                input_filename=input_path.name,
                metrics=metrics,
                params={
                    "dpi": self.config.get("preprocessing", {}).get("dpi", 300),
                    "compiler": self.config.get("validation", {}).get("compiler", "xelatex"),
                    "vlm_model": self.config.get("ollama", {}).get("models", {}).get("vlm", ""),
                    "llm_model": self.config.get("ollama", {}).get("models", {}).get("llm", ""),
                },
                success=True,
            )

            logger.info(
                "Pipeline complete: task=%s, time=%.1fs, stages=%s",
                task_id, elapsed, status.stage_times,
            )
            return status

        except Exception as e:
            status.stage = PipelineStage.FAILED
            status.error = str(e)
            status.message = f"Pipeline failed: {e}"
            self._report_progress(status)
            logger.error("Pipeline failed: %s", e, exc_info=True)
            self.scheduler.release()

            # Log failure to MLflow
            log_conversion_run(
                task_id=task_id,
                input_filename=input_path.name,
                metrics={"total_time_s": round(time.time() - start_time, 2)},
                success=False,
                tags={"error": str(e)[:250]},
            )

            return status

    def _preprocess(self, input_path: Path, output_dir: Path) -> List[tuple]:
        """Preprocess input files into enhanced page images.

        PDF-to-image conversion runs serially (single ``fitz.open`` call per
        PDF), but per-page enhancement (denoise + deskew) is parallelised with
        a :class:`~concurrent.futures.ThreadPoolExecutor`.

        Args:
            input_path: Input file or directory.
            output_dir: Output directory for intermediate files.

        Returns:
            List of (image_path, image_array) tuples, ordered by page number.
        """
        pages_dir = output_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        if input_path.is_dir():
            files = self.batch_loader.scan_directory(input_path)
            pdfs, images = self.batch_loader.classify_files(files)
        elif input_path.suffix.lower() == ".pdf":
            pdfs = [input_path]
            images = []
        else:
            pdfs = []
            images = [input_path]

        # Collect all raw (path, image) pairs first
        raw_pages: List[Tuple[Path, np.ndarray]] = []

        for pdf_path in pdfs:
            img_paths = self.pdf_converter.convert(pdf_path, pages_dir)
            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                if img is not None:
                    raw_pages.append((img_path, img))

        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                out_path = pages_dir / img_path.name
                raw_pages.append((out_path, img))

        # Enhance pages in parallel (each page is independent)
        max_workers = min(os.cpu_count() or 1, 4)

        def _enhance_page(item: Tuple[Path, np.ndarray]) -> Tuple[Path, np.ndarray]:
            img_path, img = item
            enhanced = self.image_enhancer.enhance(img, denoise=True, deskew=True)
            cv2.imwrite(str(img_path), enhanced)
            return (img_path, enhanced)

        page_images: List[Tuple[Path, np.ndarray]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_enhance_page, item) for item in raw_pages]
            for future in concurrent.futures.as_completed(futures):
                try:
                    page_images.append(future.result())
                except Exception as e:
                    logger.warning("Page enhancement failed: %s", e)

        # Re-sort by filename to preserve page order
        page_images.sort(key=lambda x: str(x[0]))
        return page_images

    def _recognize_page(
        self,
        page: Dict[str, Any],
        output_dir: Path,
        use_mcts: bool,
    ) -> List[Dict[str, Any]]:
        """Recognize all regions on a page, batched by model type.

        Regions are grouped by type to minimize expensive model switches:
        1. TEXT / TITLE / TABLE — PaddleOCR (no LLM model needed)
        2. FORMULA — Ollama VLM
        3. FIGURE — DeTikZify (or VLM fallback)
        Results are reassembled in original reading order.

        Args:
            page: Page dict with 'image', 'regions', etc.
            output_dir: Output directory for figure images.
            use_mcts: Whether to use MCTS for figures.

        Returns:
            List of recognized region dicts with 'type', 'latex', 'bbox'.
        """
        image = page["image"]
        regions = page["regions"]
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # Pre-crop all regions and build index
        indexed_regions: List[Tuple[int, Any, np.ndarray]] = []
        for i, region in enumerate(regions):
            cropped = self.layout_detector.crop_region(image, region)
            indexed_regions.append((i, region, cropped))

        # Results dict keyed by original index
        results: Dict[int, str] = {}

        # --- Batch 1: TEXT / TITLE / TABLE (PaddleOCR, no model switch) ---
        for i, region, cropped in indexed_regions:
            if region.region_type in (RegionType.TEXT, RegionType.TITLE):
                results[i] = self.text_recognizer.recognize_to_latex(cropped)
            elif region.region_type == RegionType.TABLE:
                latex_table = self.table_recognizer.recognize(cropped)
                results[i] = latex_table if latex_table else ""
            elif region.region_type == RegionType.UNKNOWN:
                results[i] = self.text_recognizer.recognize_to_latex(cropped)

        # --- Batch 2: FORMULA (Ollama VLM) ---
        formula_regions = [(i, r, c) for i, r, c in indexed_regions
                           if r.region_type == RegionType.FORMULA]
        if formula_regions:
            self.scheduler.acquire(ModelType.OLLAMA_VLM)
            for i, region, cropped in formula_regions:
                code = self.formula_recognizer.recognize(cropped)
                results[i] = code if code else ""

        # --- Batch 3: FIGURE (DeTikZify) ---
        figure_regions = [(i, r, c) for i, r, c in indexed_regions
                          if r.region_type == RegionType.FIGURE]
        if figure_regions:
            for i, region, cropped in figure_regions:
                fig_path = figures_dir / f"fig_p{page['page_num']}_r{i}.png"
                latex = self.figure_recognizer.recognize(
                    cropped,
                    use_mcts=use_mcts,
                    save_original_path=fig_path,
                )
                results[i] = latex

        # Reassemble in original reading order
        recognized = []
        for i, region, _ in indexed_regions:
            recognized.append({
                "type": region.region_type,
                "latex": results.get(i, ""),
                "bbox": region.bbox.to_tuple(),
                "confidence": region.confidence,
            })

        return recognized

    @staticmethod
    def _generate_task_id(input_path: Path) -> str:
        """Generate a unique task ID based on file path and timestamp.

        Args:
            input_path: Input file path.

        Returns:
            Short unique ID string.
        """
        content = f"{input_path}_{time.time()}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]
