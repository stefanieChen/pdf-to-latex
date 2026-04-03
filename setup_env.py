"""Cross-platform environment detection and setup for PDF-to-LaTeX system."""

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.resolve()


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def print_status(name: str, ok: bool, detail: str = "") -> None:
    """Print a status line with checkmark or X."""
    mark = "OK" if ok else "FAIL"
    detail_str = f" ({detail})" if detail else ""
    print(f"  [{mark}] {name}{detail_str}")


def get_python_cmd() -> str:
    """Get the correct python command for the current platform."""
    if platform.system() == "Windows":
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"

    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def check_python() -> bool:
    """Check Python version (3.11+ required)."""
    version = sys.version_info
    ok = version >= (3, 11)
    print_status("Python", ok, f"{version.major}.{version.minor}.{version.micro}")
    return ok


def check_gpu() -> dict:
    """Check GPU availability."""
    info = {"available": False, "type": "none", "name": ""}

    # Check NVIDIA CUDA
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            info["available"] = True
            info["type"] = "cuda"
            info["name"] = result.stdout.strip()
            print_status("GPU (CUDA)", True, info["name"])
            return info
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check Apple Metal (macOS)
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType"],
                capture_output=True, text=True, timeout=10,
            )
            if "Apple" in result.stdout or "Metal" in result.stdout:
                info["available"] = True
                info["type"] = "metal"
                info["name"] = "Apple Silicon (Metal)"
                print_status("GPU (Metal)", True, info["name"])
                return info
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    print_status("GPU", False, "No CUDA or Metal GPU detected")
    return info


def check_ollama() -> bool:
    """Check if Ollama is installed and running."""
    cmd = shutil.which("ollama")
    if not cmd:
        print_status("Ollama", False, "Not found on PATH")
        return False

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            models = [l.split()[0] for l in result.stdout.strip().split("\n")[1:] if l.strip()]
            print_status("Ollama", True, f"{len(models)} models available")
            return True
    except (subprocess.TimeoutExpired, Exception):
        pass

    print_status("Ollama", False, "Installed but not responding (run 'ollama serve')")
    return False


def check_ollama_models() -> dict:
    """Check which required Ollama models are available."""
    required = {"minicpm-v": False, "qwen2.5:7b": False}

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                for model in required:
                    if model in line:
                        required[model] = True
    except Exception:
        pass

    for model, available in required.items():
        print_status(f"Ollama model: {model}", available)

    return required


def check_tex() -> bool:
    """Check if a TeX distribution is installed."""
    for compiler in ["xelatex", "pdflatex", "lualatex"]:
        if shutil.which(compiler):
            print_status("TeX compiler", True, compiler)
            return True

    print_status("TeX compiler", False, "Not found (install TeX Live or MiKTeX)")
    return False


def check_ghostscript() -> bool:
    """Check if Ghostscript is installed (required by DeTikZify)."""
    for cmd in ["gs", "gswin64c", "gswin32c"]:
        if shutil.which(cmd):
            print_status("Ghostscript", True, cmd)
            return True

    print_status("Ghostscript", False, "Not found")
    return False


def check_poppler() -> bool:
    """Check if Poppler is installed."""
    for cmd in ["pdftoppm", "pdftotext"]:
        if shutil.which(cmd):
            print_status("Poppler", True, cmd)
            return True

    print_status("Poppler", False, "Not found (optional)")
    return False


def check_node() -> bool:
    """Check if Node.js is installed (for React frontend)."""
    node = shutil.which("node")
    if node:
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=5)
            print_status("Node.js", True, result.stdout.strip())
            return True
        except Exception:
            pass

    print_status("Node.js", False, "Not found (needed for web frontend)")
    return False


def check_venv() -> bool:
    """Check if the virtual environment exists."""
    if platform.system() == "Windows":
        venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    else:
        venv_python = PROJECT_ROOT / "venv" / "bin" / "python"

    ok = venv_python.exists()
    print_status("Virtual env (venv)", ok, str(PROJECT_ROOT / "venv") if ok else "Not created")
    return ok


def run_check() -> None:
    """Run all environment checks."""
    print_header("PDF-to-LaTeX Environment Check")
    print(f"  Platform: {platform.platform()}")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Architecture: {platform.machine()}")

    print_header("Core Requirements")
    check_python()
    check_venv()
    check_gpu()

    print_header("External Tools")
    check_ollama()
    check_ollama_models()
    check_tex()
    check_ghostscript()
    check_poppler()
    check_node()

    print(f"\n{'='*60}")
    print("  Check complete. Fix any [FAIL] items above before proceeding.")
    print(f"{'='*60}\n")


def run_install() -> None:
    """Install Python dependencies and pull required models."""
    print_header("Installing Dependencies")

    python_cmd = get_python_cmd()

    # Install Python packages
    print("\n  [1/3] Installing Python packages...")
    req_file = PROJECT_ROOT / "requirements.txt"
    
    # Use Tsinghua mirror for faster downloads in China
    mirror_index = "https://pypi.tuna.tsinghua.edu.cn/simple"
    mirror_host = "pypi.tuna.tsinghua.edu.cn"
    
    # Install core packages with mirror
    subprocess.run(
        [python_cmd, "-m", "pip", "install", "-r", str(req_file), 
         "-i", mirror_index, "--trusted-host", mirror_host],
        cwd=str(PROJECT_ROOT),
    )
    
    # Special handling for heavy packages (platform-specific mirrors)
    print("\n  Installing heavy dependencies with optimized mirrors...")
    
    # PaddlePaddle - platform-specific official mirror
    print("    Installing PaddlePaddle...")
    if platform.system() == "Darwin":
        paddle_find_url = "https://www.paddlepaddle.org.cn/whl/mac/cpu/stable.html"
    else:
        paddle_find_url = "https://www.paddlepaddle.org.cn/whl/windows/cpu-mkl-avx/stable.html"
    subprocess.run(
        [python_cmd, "-m", "pip", "install", "paddlepaddle>=2.5.0",
         "-f", paddle_find_url,
         "-i", mirror_index, "--trusted-host", mirror_host],
        cwd=str(PROJECT_ROOT),
    )
    
    # PaddleOCR - use mirror
    print("    Installing PaddleOCR...")
    subprocess.run(
        [python_cmd, "-m", "pip", "install", "paddleocr>=2.7.0",
         "-i", mirror_index, "--trusted-host", mirror_host],
        cwd=str(PROJECT_ROOT),
    )
    
    # pix2tex - use mirror
    print("    Installing pix2tex...")
    subprocess.run(
        [python_cmd, "-m", "pip", "install", "pix2tex>=0.1.1",
         "-i", mirror_index, "--trusted-host", mirror_host],
        cwd=str(PROJECT_ROOT),
    )
    
    # DeTikZify - use GitHub proxy for China network
    print("    Installing DeTikZify...")
    subprocess.run(
        [python_cmd, "-m", "pip", "install",
         "detikzify @ git+https://ghproxy.com/https://github.com/potamides/DeTikZify"],
        cwd=str(PROJECT_ROOT),
    )

    # Pull Ollama models
    print("\n  [2/3] Pulling Ollama models...")
    for model in ["minicpm-v", "qwen2.5:7b"]:
        print(f"    Pulling {model}...")
        try:
            subprocess.run(["ollama", "pull", model], timeout=1800)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            print(f"    Warning: Could not pull {model}: {e}")

    # Create directories
    print("\n  [3/3] Creating directories...")
    for d in ["data/input", "data/output", "data/temp", "data/test_samples", "logs"]:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)

    print_header("Installation Complete")
    print("  Run 'python setup_env.py --check' to verify.")
    print("\n  NOTE: If installation failed, try running:")
    print("    python setup_env.py --install")


def main():
    """Main entry point for environment setup."""
    parser = argparse.ArgumentParser(description="PDF-to-LaTeX Environment Setup")
    parser.add_argument("--check", action="store_true", help="Check environment readiness")
    parser.add_argument("--install", action="store_true", help="Install dependencies")
    args = parser.parse_args()

    if args.check:
        run_check()
    elif args.install:
        run_install()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
