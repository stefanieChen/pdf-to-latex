# PDF-to-LaTeX Conversion System

将 PDF（含扫描件）或批量图像文件完整复刻为 LaTeX 代码，保留原始布局、中英文文字、数学公式、表格和几何图形。

## Features

- **中英文 OCR** — PaddleOCR 驱动，支持混排文档
- **数学公式识别** — pix2tex + VLM 双引擎
- **表格识别** — PPStructure 表格提取 → LaTeX tabular
- **图形→TikZ** — DeTikZify (NeurIPS 2024) + MCTS 迭代优化
- **版面还原** — 自动检测多栏布局、阅读顺序
- **自动校验** — xelatex 编译验证 + SSIM 视觉对比 + LLM 审核
- **Web 界面** — FastAPI 后端 + 静态前端，实时进度 WebSocket
- **跨平台** — 支持 Windows 和 macOS

## Quick Start

### 1. Prerequisites

- **Python 3.11+**（DeTikZify 要求）
- **Ollama** with models: `minicpm-v`, `qwen2.5:7b`
- **TeX Live** or **MiKTeX** (for xelatex)
- **Ghostscript** + **Poppler** (for DeTikZify)

macOS 系统依赖可通过 Homebrew 一键安装：

```bash
brew install ollama ghostscript poppler
brew install --cask mactex
```

### 2. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install all dependencies (uses Chinese mirrors)
python setup_env.py --install

# Verify environment
python setup_env.py --check
```

> **Note:** 重型依赖（PaddlePaddle、PaddleOCR、pix2tex、DeTikZify）由 `setup_env.py --install` 统一安装，使用国内镜像，无需手动 pip install。

### 3. Start the System

```bash
python start.py
```

This starts:
- Ollama server (if not running)
- API server at `http://localhost:8000`
- Static frontend at `http://localhost:8000`

### 4. API Usage

```bash
# Upload a file
curl -X POST http://localhost:8000/api/upload -F "file=@document.pdf"

# Check status
curl http://localhost:8000/api/task/{task_id}/status

# Get LaTeX result
curl http://localhost:8000/api/task/{task_id}/result

# Download all outputs as ZIP
curl -O http://localhost:8000/api/task/{task_id}/download
```

API documentation: `http://localhost:8000/docs`

## Deployment on a New Machine

The project is designed for easy deployment via `git clone` on both **Windows** and **macOS**.

### macOS (Apple Silicon / Intel)

```bash
# 1. Clone the repo
git clone <repo-url> pdf-to-latex && cd pdf-to-latex

# 2. Install system dependencies via Homebrew
brew install ollama ghostscript poppler
brew install --cask mactex-no-gui

# 3. Create venv and install
python3.11 -m venv venv
source venv/bin/activate
python setup_env.py --install

# 4. Verify
python setup_env.py --check

# 5. Start
python start.py
```

### Windows

```bash
# 1. Clone the repo
git clone <repo-url> pdf-to-latex && cd pdf-to-latex

# 2. Install system dependencies:
#    - Ollama: https://ollama.com/download
#    - MiKTeX: https://miktex.org/download
#    - Ghostscript: https://ghostscript.com/releases/

# 3. Create venv and install
python -m venv venv
venv\Scripts\activate
python setup_env.py --install

# 4. Verify
python setup_env.py --check

# 5. Start
python start.py
```

## Project Structure

```
pdf2latex/
├── config/
│   ├── settings.yaml           # Global configuration
│   └── latex_templates/        # LaTeX templates
├── src/
│   ├── preprocessing/          # PDF→image, enhancement, native text extraction
│   ├── layout/                 # Layout detection
│   ├── recognition/            # OCR, formula, table, figure (batched by model)
│   ├── assembly/               # LaTeX assembly
│   ├── validation/             # Compilation, pre-validation, visual comparison
│   ├── monitoring/             # MLflow integration
│   ├── config.py               # Config loader
│   ├── ollama_client.py        # Ollama API client (keep_alive, in-memory images)
│   ├── detikzify_client.py     # DeTikZify wrapper
│   ├── model_scheduler.py      # VRAM management
│   ├── task_store.py           # SQLite task persistence
│   └── pipeline.py             # Main pipeline (parallel, batched, instrumented)
├── scripts/
│   ├── benchmark.py            # Performance benchmark runner
│   ├── manage_chromadb.py
│   └── start_monitoring.py
├── tests/                      # Unit tests
├── server.py                   # FastAPI backend
├── setup_env.py                # Cross-platform environment setup
├── start.py                    # One-click start
└── requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

## Performance Benchmarking

Run the pipeline on test documents and collect per-stage timing, SSIM scores, and memory usage:

```bash
# Run benchmark on test samples
python -m scripts.benchmark --input data/test_samples

# Save results as a baseline for future comparison
python -m scripts.benchmark --input data/test_samples --save-baseline

# Compare against a saved baseline
python -m scripts.benchmark --input data/test_samples --compare data/benchmark_output/baseline.json
```

## Performance Optimizations

The following optimizations are implemented to reduce end-to-end latency:

| Optimization | Expected Impact |
|---|---|
| **Batch regions by model type** — TEXT/TABLE first, then FORMULA, then FIGURE | ~50-80% fewer model switches |
| **Parallel page preprocessing** — ThreadPoolExecutor for image enhancement | Near-linear speedup on multi-page PDFs |
| **Noise-aware denoising** — skip/bilateral for clean images, NLM only for noisy | 3-10x faster per clean page |
| **In-memory image passing** — numpy arrays passed directly to Ollama (no temp files) | ~50-200 ms saved per region |
| **Ollama keep_alive** — models stay loaded for 5 min between calls | Eliminates redundant model reloads |
| **Progressive DeTikZify** — fast sample() first, MCTS only if SSIM < 0.7 | Avoids 5-min MCTS for simple figures |
| **LaTeX pre-validation** — catch syntax errors before spawning xelatex | Skip 60 s compile for broken code |
| **Math-aware error fixer** — tracks math mode context when escaping chars | Fewer fix-loop iterations |
| **SQLite task persistence** — replaces in-memory dict | Survives restarts, bounded memory |
| **ZIP result caching** — cached after first build, invalidated on recompile | Instant repeated downloads |
| **Engine warm-up at startup** — PaddleOCR engines pre-initialized | No first-request cold start |

## Monitoring & Observability

### MLflow (Pipeline Metrics Tracking)

Track conversion metrics (latency, per-stage timing, SSIM, compilation success) across runs.

```bash
# Install
pip install mlflow

# Enable in config/settings.yaml
# monitoring.mlflow.enabled: true

# Start MLflow UI
mlflow server --host 127.0.0.1 --port 5000
# Open http://localhost:5000
```

Metrics logged per conversion: `total_time_s`, `stage_preprocessing_s`, `stage_layout_analysis_s`, `stage_recognition_s`, `stage_assembly_s`, `stage_validation_s`, `ssim_score`, `compilation_success`, `num_pages`, `latex_length`.

## Architecture

```
Input(PDF/Images) → Preprocessing → Layout Analysis → Recognition → Assembly → Validation → Output(.tex)
                    (parallel)       (per-page)        (batched by    (template)  (pre-validate
                                                        model type)               + compile + SSIM)
```

**Model Management (8GB VRAM):**
- DeTikZify (4-bit quantized ~5GB) for figure→TikZ
- minicpm-v (Ollama, keep_alive=5m) for VLM tasks
- qwen2.5:7b (Ollama, keep_alive=5m) for LaTeX code fixing
- Models loaded/unloaded via `ModelScheduler`, minimized by batching regions
