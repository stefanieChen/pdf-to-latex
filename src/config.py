"""Configuration loader for the PDF-to-LaTeX system."""

import logging
import os
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ── PaddlePaddle 3.x compatibility workarounds (Windows & macOS) ──────────────
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

# Pre-import torch before paddle to prevent DLL loading conflict on Windows.
# Paddle's DLL initialization corrupts torch's ability to load shm.dll.
if platform.system() == "Windows":
    try:
        import torch  # noqa: F401
    except ImportError:
        pass

# Disable PIR (new_ir) executor on CPU to avoid
# "ConvertPirAttribute2RuntimeAttribute not support" crash in PaddlePaddle 3.x.
# The crash is in the new executor's oneDNN instruction handler. We patch the
# paddlex source file directly because:
#   - enable_new_ir is controlled at the C++ paddle.inference.Config level
#   - Python-level monkey-patches on PaddlePredictorOption do NOT prevent it
# CRITICAL: We locate the file via importlib.util.find_spec (no code execution)
# so the patch is applied BEFORE paddlex is ever imported.
def _patch_paddlex_cpu_inference() -> None:
    """Patch static_infer.py to force enable_new_ir(False) on CPU."""
    import importlib.util

    spec = importlib.util.find_spec("paddlex")
    if spec is None or spec.submodule_search_locations is None:
        return

    # Locate the file without importing it
    paddlex_root = Path(list(spec.submodule_search_locations)[0])
    src_path = paddlex_root / "inference" / "models" / "common" / "static_infer.py"
    if not src_path.exists():
        return

    _ORIGINAL = "config.enable_new_ir(self._option.enable_new_ir)"
    _PATCHED = "config.enable_new_ir(False)"

    try:
        content = src_path.read_text(encoding="utf-8")
    except OSError:
        return

    if _ORIGINAL not in content:
        return  # already patched or different version

    content = content.replace(_ORIGINAL, _PATCHED)
    try:
        src_path.write_text(content, encoding="utf-8")
        # Remove stale .pyc so Python loads the patched source
        import shutil
        pyc_dir = src_path.parent / "__pycache__"
        if pyc_dir.exists():
            shutil.rmtree(pyc_dir, ignore_errors=True)
    except OSError:
        pass

_patch_paddlex_cpu_inference()
# ── End workarounds ──────────────────────────────────────────────────────────


# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/settings.yaml.

    Returns:
        Configuration dictionary.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "settings.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute
    for key in ["input_dir", "output_dir", "temp_dir", "log_dir"]:
        if key in config.get("paths", {}):
            config["paths"][key] = str(PROJECT_ROOT / config["paths"][key])

    # Ensure directories exist
    for dir_path in config.get("paths", {}).values():
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    return config


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Set up logging based on configuration.

    Args:
        config: Configuration dictionary.

    Returns:
        Root logger instance.
    """
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    log_file = log_config.get("file")

    if log_file:
        log_file_path = Path(log_file)
        if not log_file_path.is_absolute():
            log_file_path = PROJECT_ROOT / log_file_path
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = str(log_file_path)

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)
    logger = logging.getLogger("pdf2latex")
    logger.setLevel(log_level)
    return logger


def get_paddle_device(use_gpu: bool = True) -> str:
    """Return the PaddlePaddle device string with GPU auto-detection.

    Args:
        use_gpu: Whether GPU is requested.  When True, returns "gpu" if
            PaddlePaddle was compiled with CUDA and a GPU is present;
            otherwise falls back to "cpu".

    Returns:
        "gpu" or "cpu".
    """
    if not use_gpu:
        return "cpu"
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:
        pass
    return "cpu"


def get_platform_info() -> Dict[str, str]:
    """Get current platform information.

    Returns:
        Dictionary with OS, architecture, and Python version.
    """
    return {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
