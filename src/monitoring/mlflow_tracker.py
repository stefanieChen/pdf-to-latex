"""MLflow integration for PDF-to-LaTeX pipeline metrics tracking.

Logs conversion metrics (latency, SSIM, compilation success) to a
local MLflow tracking server.

Usage:
    from src.monitoring.mlflow_tracker import init_mlflow, log_conversion_run

    init_mlflow(config)
    log_conversion_run(task_id="abc123", metrics={...}, params={...})
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("pdf2latex.monitoring.mlflow")

_MLFLOW_INITIALIZED = False


def is_mlflow_available() -> bool:
    """Check whether the mlflow package is installed.

    Returns:
        True if mlflow is importable.
    """
    try:
        import mlflow  # noqa: F401
        return True
    except ImportError:
        return False


def init_mlflow(config: Optional[Dict[str, Any]] = None) -> bool:
    """Initialize MLflow tracking with local configuration.

    Args:
        config: Configuration dictionary. Reads ``monitoring.mlflow``
                section for ``enabled``, ``tracking_uri``, and
                ``experiment_name``.

    Returns:
        True if MLflow was successfully initialized.
    """
    global _MLFLOW_INITIALIZED
    if _MLFLOW_INITIALIZED:
        return True

    if config is None:
        from src.config import load_config
        config = load_config()

    mon_cfg = config.get("monitoring", {}).get("mlflow", {})
    enabled = mon_cfg.get("enabled", False)

    if not enabled:
        logger.info("MLflow tracking is disabled in config")
        return False

    if not is_mlflow_available():
        logger.warning(
            "MLflow enabled in config but not installed. "
            "Run: pip install mlflow"
        )
        return False

    tracking_uri = mon_cfg.get("tracking_uri", "mlruns")
    experiment_name = mon_cfg.get("experiment_name", "pdf-to-latex")

    try:
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        _MLFLOW_INITIALIZED = True
        logger.info(
            f"MLflow initialized: uri={tracking_uri}, "
            f"experiment={experiment_name}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}")
        return False


def log_conversion_run(
    task_id: str,
    input_filename: str,
    metrics: Optional[Dict[str, float]] = None,
    params: Optional[Dict[str, Any]] = None,
    success: bool = True,
    tags: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Log a PDF-to-LaTeX conversion run to MLflow.

    Args:
        task_id: Unique task identifier.
        input_filename: Original input filename.
        metrics: Metric name → value mapping (e.g., latency, SSIM, pages).
        params: Pipeline parameters to log.
        success: Whether the conversion completed successfully.
        tags: Additional tags.

    Returns:
        MLflow run ID if successful, None otherwise.
    """
    if not _MLFLOW_INITIALIZED:
        return None

    try:
        import mlflow

        with mlflow.start_run(run_name=f"convert_{task_id}") as run:
            mlflow.log_param("task_id", task_id)
            mlflow.log_param("input_filename", input_filename[:250])
            mlflow.log_param("success", success)

            if params:
                for key, value in params.items():
                    mlflow.log_param(key, str(value)[:250])

            if metrics:
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(metric_name, value)

            mlflow.set_tag("task.id", task_id)
            mlflow.set_tag("task.success", str(success))
            if tags:
                for tag_key, tag_val in tags.items():
                    mlflow.set_tag(tag_key, tag_val)

            logger.info(
                f"MLflow conversion run logged: {run.info.run_id} "
                f"(task={task_id}, success={success})"
            )
            return run.info.run_id

    except Exception as e:
        logger.error(f"Failed to log MLflow conversion run: {e}")
        return None
