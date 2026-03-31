"""Configuration loader for the PDF-to-LaTeX system."""

import logging
import platform
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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
