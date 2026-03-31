"""FastAPI backend server for the PDF-to-LaTeX conversion system."""

import asyncio
import hashlib
import logging
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import load_config, setup_logging, PROJECT_ROOT
from src.pipeline import Pipeline, PipelineStage, TaskStatus

STATIC_DIR = PROJECT_ROOT / "static"

logger = logging.getLogger("pdf2latex.server")

# Load config and setup logging
config = load_config()
setup_logging(config)

app = FastAPI(
    title="PDF-to-LaTeX Conversion API",
    description="Convert PDF and image files to LaTeX code preserving layout, text, formulas, tables, and figures.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Task storage
tasks: Dict[str, Dict[str, Any]] = {}
task_websockets: Dict[str, list] = {}

# Pipeline instance (lazy init)
_pipeline: Optional[Pipeline] = None


def get_pipeline() -> Pipeline:
    """Get or create the pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline(config)
    return _pipeline


def get_upload_dir() -> Path:
    """Get the upload directory path."""
    upload_dir = Path(config.get("paths", {}).get("input_dir", "data/input"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


def get_output_dir() -> Path:
    """Get the output directory path."""
    output_dir = Path(config.get("paths", {}).get("output_dir", "data/output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# --- Frontend Route ---

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main frontend page."""
    index_path = STATIC_DIR / "index.html"
    return index_path.read_text(encoding="utf-8")


# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": time.time()}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file for conversion.

    Args:
        file: Uploaded PDF or image file.

    Returns:
        JSON with task_id for tracking the conversion.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file extension
    suffix = Path(file.filename).suffix.lower()
    supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
    if suffix not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Supported: {', '.join(supported)}",
        )

    # Check file size
    max_size = config.get("server", {}).get("upload_max_size_mb", 100) * 1024 * 1024
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="File too large")

    # Generate task ID
    task_id = hashlib.sha256(f"{file.filename}_{time.time()}_{uuid.uuid4()}".encode()).hexdigest()[:12]

    # Save uploaded file
    upload_dir = get_upload_dir() / task_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / file.filename
    file_path.write_bytes(content)

    # Initialize task
    tasks[task_id] = {
        "id": task_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "stage": PipelineStage.INIT.value,
        "progress": 0.0,
        "message": "File uploaded, waiting to process",
        "error": None,
        "output_tex": None,
        "output_pdf": None,
        "created_at": time.time(),
    }

    # Start conversion in background
    asyncio.create_task(_run_conversion(task_id, file_path))

    logger.info("Task created: %s for file %s", task_id, file.filename)
    return {"task_id": task_id, "filename": file.filename}


@app.get("/api/task/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the current status of a conversion task.

    Args:
        task_id: Task identifier.

    Returns:
        JSON with current task status.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


@app.get("/api/task/{task_id}/result")
async def get_task_result(task_id: str):
    """Get the LaTeX code result for a completed task.

    Args:
        task_id: Task identifier.

    Returns:
        JSON with LaTeX code.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    tex_path = task.get("output_tex")
    if not tex_path or not Path(tex_path).exists():
        raise HTTPException(status_code=404, detail="Result not available yet")

    latex_code = Path(tex_path).read_text(encoding="utf-8")
    return {"task_id": task_id, "latex": latex_code}


@app.get("/api/task/{task_id}/pdf")
async def get_task_pdf(task_id: str):
    """Download the compiled PDF for a completed task.

    Args:
        task_id: Task identifier.

    Returns:
        PDF file response.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    pdf_path = task.get("output_pdf")
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(status_code=404, detail="PDF not available")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"{task_id}_output.pdf",
    )


@app.get("/api/task/{task_id}/download")
async def download_task_zip(task_id: str):
    """Download all output files as a ZIP archive.

    Args:
        task_id: Task identifier.

    Returns:
        ZIP file response.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = tasks[task_id]
    output_dir = get_output_dir() / task_id
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output not available")

    zip_path = output_dir / f"{task_id}_output.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.rglob("*"):
            if file.is_file() and file != zip_path:
                zf.write(file, file.relative_to(output_dir))

    return FileResponse(
        path=str(zip_path),
        media_type="application/zip",
        filename=f"{task_id}_output.zip",
    )


@app.post("/api/task/{task_id}/recompile")
async def recompile_task(task_id: str, latex: dict):
    """Recompile with user-modified LaTeX code.

    Args:
        task_id: Task identifier.
        latex: JSON body with 'code' field containing LaTeX source.

    Returns:
        JSON with compilation status.
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    code = latex.get("code", "")
    if not code:
        raise HTTPException(status_code=400, detail="No LaTeX code provided")

    output_dir = get_output_dir() / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = get_pipeline()
    tex_path = output_dir / "output.tex"
    tex_path.write_text(code, encoding="utf-8")

    success, log_output, pdf_path = pipeline.compiler.compile(tex_path, output_dir)

    tasks[task_id]["output_tex"] = str(tex_path)
    if success and pdf_path:
        tasks[task_id]["output_pdf"] = str(pdf_path)

    return {
        "task_id": task_id,
        "success": success,
        "log": log_output[:2000] if log_output else "",
        "pdf_available": success,
    }


@app.get("/api/models/status")
async def get_models_status():
    """Check availability of required models.

    Returns:
        JSON with model status information.
    """
    pipeline = get_pipeline()

    ollama_available = pipeline.ollama_client.is_available()
    models = []
    if ollama_available:
        models = pipeline.ollama_client.list_models()

    model_names = [m.get("name", "") for m in models]

    return {
        "ollama_available": ollama_available,
        "ollama_models": model_names,
        "detikzify_loaded": pipeline.detikzify_client.is_loaded,
        "latex_compiler_available": pipeline.compiler.is_available(),
    }


# --- WebSocket ---

@app.websocket("/ws/task/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time task progress updates.

    Args:
        websocket: WebSocket connection.
        task_id: Task identifier to track.
    """
    await websocket.accept()

    if task_id not in task_websockets:
        task_websockets[task_id] = []
    task_websockets[task_id].append(websocket)

    try:
        # Send current status immediately
        if task_id in tasks:
            await websocket.send_json(tasks[task_id])

        # Keep connection alive until task completes or client disconnects
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                # Send heartbeat
                if task_id in tasks:
                    await websocket.send_json(tasks[task_id])
                    if tasks[task_id].get("stage") in (
                        PipelineStage.COMPLETE.value,
                        PipelineStage.FAILED.value,
                    ):
                        break
    except WebSocketDisconnect:
        pass
    finally:
        if task_id in task_websockets:
            task_websockets[task_id].remove(websocket)


# --- Background Task ---

async def _run_conversion(task_id: str, file_path: Path) -> None:
    """Run the conversion pipeline in the background.

    Args:
        task_id: Task identifier.
        file_path: Path to the uploaded file.
    """
    pipeline = get_pipeline()
    output_dir = get_output_dir() / task_id

    def progress_callback(status: TaskStatus):
        """Update task status and notify WebSocket clients."""
        tasks[task_id].update({
            "stage": status.stage.value,
            "progress": status.progress,
            "message": status.message,
            "error": status.error,
            "output_tex": str(status.output_tex) if status.output_tex else None,
            "output_pdf": str(status.output_pdf) if status.output_pdf else None,
        })
        # Notify WebSocket clients
        asyncio.create_task(_notify_websockets(task_id))

    pipeline.set_progress_callback(progress_callback)

    # Run in executor to avoid blocking event loop
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.convert(file_path, output_dir, task_id),
    )

    # Final status update
    tasks[task_id].update({
        "stage": result.stage.value,
        "progress": result.progress,
        "message": result.message,
        "error": result.error,
        "output_tex": str(result.output_tex) if result.output_tex else None,
        "output_pdf": str(result.output_pdf) if result.output_pdf else None,
    })
    await _notify_websockets(task_id)


async def _notify_websockets(task_id: str) -> None:
    """Send status update to all connected WebSocket clients for a task.

    Args:
        task_id: Task identifier.
    """
    if task_id not in task_websockets:
        return

    status = tasks.get(task_id, {})
    dead = []

    for ws in task_websockets[task_id]:
        try:
            await ws.send_json(status)
        except Exception:
            dead.append(ws)

    for ws in dead:
        task_websockets[task_id].remove(ws)


if __name__ == "__main__":
    import uvicorn
    server_cfg = config.get("server", {})
    uvicorn.run(
        "server:app",
        host=server_cfg.get("host", "0.0.0.0"),
        port=server_cfg.get("port", 8000),
        reload=False,
    )
