# PDF-to-LaTeX Conversion System

将 PDF（含扫描件）或批量图像文件完整复刻为 LaTeX 代码，保留原始布局、中英文文字、数学公式、表格和几何图形。

## Features

- **中英文 OCR** — PaddleOCR 驱动，支持混排文档
- **数学公式识别** — pix2tex + VLM 双引擎
- **表格识别** — PPStructure 表格提取 → LaTeX tabular
- **图形→TikZ** — DeTikZify (NeurIPS 2024) + MCTS 迭代优化
- **版面还原** — 自动检测多栏布局、阅读顺序
- **自动校验** — xelatex 编译验证 + SSIM 视觉对比 + LLM 审核
- **Web 界面** — FastAPI 后端 + React 前端，实时进度 WebSocket
- **跨平台** — 支持 Windows 和 macOS

## Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Check environment
python setup_env.py --check

# Install dependencies
python setup_env.py --install
```

### 2. Prerequisites

- **Python** 3.8+
- **Ollama** with models: `minicpm-v`, `qwen2.5:7b`
- **TeX Live** or **MiKTeX** (for xelatex)
- **Ghostscript** + **Poppler** (for DeTikZify)

### 3. Start the System

```bash
python start.py
```

This starts:
- Ollama server (if not running)
- API server at `http://localhost:8000`
- Frontend at `http://localhost:3000` (if web/ exists)

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

## Project Structure

```
pdf2latex/
├── config/
│   ├── settings.yaml           # Global configuration
│   └── latex_templates/        # LaTeX templates
├── src/
│   ├── preprocessing/          # PDF→image, enhancement
│   ├── layout/                 # Layout detection
│   ├── recognition/            # OCR, formula, table, figure
│   ├── assembly/               # LaTeX assembly
│   ├── validation/             # Compilation, visual comparison
│   ├── config.py               # Config loader
│   ├── ollama_client.py        # Ollama API client
│   ├── detikzify_client.py     # DeTikZify wrapper
│   ├── model_scheduler.py      # VRAM management
│   └── pipeline.py             # Main pipeline
├── tests/                      # Unit tests
├── server.py                   # FastAPI backend
├── setup_env.py                # Environment setup
├── start.py                    # One-click start
└── requirements.txt
```

## Testing

```bash
pytest tests/ -v
```

## Monitoring & Observability

### MLflow (Pipeline Metrics Tracking)

Track conversion metrics (latency, SSIM, compilation success) across runs.

```bash
# Install
pip install mlflow

# Enable in config/settings.yaml
# monitoring.mlflow.enabled: true

# Start MLflow UI
mlflow server --host 127.0.0.1 --port 5000
# Open http://localhost:5000
```

## Architecture

```
Input(PDF/Images) → Preprocessing → Layout Analysis → Recognition → Assembly → Validation → Output(.tex)
```

**Model Management (8GB VRAM):**
- DeTikZify (4-bit quantized ~5GB) for figure→TikZ
- minicpm-v (Ollama) for VLM tasks
- qwen2.5:7b (Ollama) for LaTeX code fixing
- Models loaded/unloaded serially via `ModelScheduler`
