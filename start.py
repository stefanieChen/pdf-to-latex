"""One-click start script for the PDF-to-LaTeX system."""

import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()


def get_python_cmd() -> str:
    """Get the Python command from venv."""
    if platform.system() == "Windows":
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def check_ollama_running() -> bool:
    """Check if Ollama server is running."""
    try:
        import httpx
        resp = httpx.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def start_ollama() -> subprocess.Popen:
    """Start Ollama server if not already running."""
    if check_ollama_running():
        print("  Ollama server already running")
        return None

    print("  Starting Ollama server...")
    if platform.system() == "Windows":
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        proc = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )

    # Wait for startup
    for _ in range(30):
        time.sleep(1)
        if check_ollama_running():
            print("  Ollama server started")
            return proc

    print("  Warning: Ollama server may not have started properly")
    return proc


def start_api_server() -> subprocess.Popen:
    """Start the FastAPI backend server."""
    python_cmd = get_python_cmd()
    print("  Starting API server on http://localhost:8000 ...")

    proc = subprocess.Popen(
        [python_cmd, "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(PROJECT_ROOT),
    )

    # Wait briefly for startup
    time.sleep(2)
    return proc


def start_frontend() -> subprocess.Popen:
    """Start the React frontend dev server if web/ directory exists."""
    web_dir = PROJECT_ROOT / "web"
    if not web_dir.exists():
        print("  Frontend (web/) not found, skipping")
        return None

    package_json = web_dir / "package.json"
    if not package_json.exists():
        print("  No package.json in web/, skipping frontend")
        return None

    # Check if node_modules exists
    node_modules = web_dir / "node_modules"
    if not node_modules.exists():
        print("  Installing frontend dependencies...")
        subprocess.run(["npm", "install"], cwd=str(web_dir))

    print("  Starting frontend on http://localhost:3000 ...")
    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(web_dir),
    )
    return proc


def main():
    """Start all services."""
    print("\n" + "=" * 50)
    print("  PDF-to-LaTeX System Startup")
    print("=" * 50)

    processes = []

    try:
        # 1. Ollama
        print("\n[1/3] Ollama Server")
        ollama_proc = start_ollama()
        if ollama_proc:
            processes.append(ollama_proc)

        # 2. API Server
        print("\n[2/3] API Server")
        api_proc = start_api_server()
        processes.append(api_proc)

        # 3. Frontend
        print("\n[3/3] Frontend")
        fe_proc = start_frontend()
        if fe_proc:
            processes.append(fe_proc)

        print("\n" + "=" * 50)
        print("  All services started!")
        print("  API:      http://localhost:8000")
        print("  API Docs: http://localhost:8000/docs")
        if fe_proc:
            print("  Frontend: http://localhost:3000")
        print("  Press Ctrl+C to stop all services")
        print("=" * 50 + "\n")

        # Wait for processes
        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        for proc in processes:
            try:
                if platform.system() == "Windows":
                    proc.terminate()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # Wait for graceful shutdown
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print("All services stopped.")


if __name__ == "__main__":
    main()
