"""Benchmark runner for the PDF-to-LaTeX pipeline.

Runs the pipeline on a set of test documents, collects per-stage timing,
SSIM scores, and memory usage, then outputs a summary table.  Results can
be compared against a saved baseline.

Usage:
    python -m scripts.benchmark --input data/test_samples
    python -m scripts.benchmark --input data/test_samples --save-baseline
    python -m scripts.benchmark --input data/test_samples --compare baseline.json

Cross-platform: works on both Windows and macOS.
"""

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, setup_logging
from src.pipeline import Pipeline, TaskStatus


logger = logging.getLogger("pdf2latex.benchmark")


def get_memory_mb() -> float:
    """Return current process RSS in megabytes (cross-platform).

    Returns:
        RSS in MB, or 0.0 if psutil is unavailable.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_gpu_memory_mb() -> float:
    """Return peak GPU memory allocated in MB (CUDA only).

    Returns:
        Peak VRAM in MB, or 0.0 if torch/CUDA unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 * 1024)
    except ImportError:
        pass
    return 0.0


def run_single(
    pipeline: Pipeline,
    input_path: Path,
    output_base: Path,
    use_mcts: bool = False,
) -> Dict[str, Any]:
    """Run the pipeline on a single file and collect metrics.

    Args:
        pipeline: Initialised Pipeline instance.
        input_path: Path to input PDF or image.
        output_base: Base output directory.
        use_mcts: Whether to enable MCTS for figures.

    Returns:
        Dict with timing, quality, and memory metrics.
    """
    mem_before = get_memory_mb()
    t0 = time.perf_counter()

    status: TaskStatus = pipeline.convert(
        input_path=input_path,
        output_dir=output_base,
        use_mcts=use_mcts,
    )

    elapsed = time.perf_counter() - t0
    mem_after = get_memory_mb()

    return {
        "file": input_path.name,
        "task_id": status.task_id,
        "success": status.stage.value == "complete",
        "total_time_s": round(elapsed, 2),
        "stage_times": status.stage_times,
        "ssim_score": status.ssim_score,
        "compilation_ok": status.output_pdf is not None,
        "ram_delta_mb": round(mem_after - mem_before, 1),
        "ram_peak_mb": round(mem_after, 1),
        "gpu_peak_mb": round(get_gpu_memory_mb(), 1),
        "error": status.error,
    }


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a formatted summary table of benchmark results.

    Args:
        results: List of per-file result dicts.
    """
    print("\n" + "=" * 90)
    print(f"{'File':<30} {'Time(s)':>8} {'Preproc':>8} {'Layout':>8} "
          f"{'Recog':>8} {'Valid':>8} {'SSIM':>6} {'OK':>4}")
    print("-" * 90)

    total_time = 0.0
    for r in results:
        st = r.get("stage_times", {})
        ssim = f"{r['ssim_score']:.3f}" if r.get("ssim_score") is not None else "  -  "
        ok = "Y" if r["success"] else "N"
        print(f"{r['file']:<30} {r['total_time_s']:>8.1f} "
              f"{st.get('preprocessing', 0):>8.1f} "
              f"{st.get('layout_analysis', 0):>8.1f} "
              f"{st.get('recognition', 0):>8.1f} "
              f"{st.get('validation', 0):>8.1f} "
              f"{ssim:>6} {ok:>4}")
        total_time += r["total_time_s"]

    print("-" * 90)
    n = len(results)
    success_count = sum(1 for r in results if r["success"])
    avg_time = total_time / n if n else 0
    ssim_vals = [r["ssim_score"] for r in results if r.get("ssim_score") is not None]
    avg_ssim = sum(ssim_vals) / len(ssim_vals) if ssim_vals else 0

    print(f"{'TOTAL':<30} {total_time:>8.1f}")
    print(f"  Files: {n}  |  Success: {success_count}/{n}  |  "
          f"Avg time: {avg_time:.1f}s  |  Avg SSIM: {avg_ssim:.3f}")
    print(f"  Platform: {platform.system()} {platform.machine()}  |  "
          f"RAM peak: {max((r['ram_peak_mb'] for r in results), default=0):.0f} MB  |  "
          f"GPU peak: {max((r['gpu_peak_mb'] for r in results), default=0):.0f} MB")
    print("=" * 90 + "\n")


def compare_baseline(
    results: List[Dict[str, Any]], baseline_path: Path,
) -> None:
    """Compare current results against a saved baseline.

    Args:
        results: Current benchmark results.
        baseline_path: Path to baseline JSON file.
    """
    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}")
        return

    with open(baseline_path, "r") as f:
        baseline = json.load(f)

    base_map = {r["file"]: r for r in baseline}

    print("\n" + "=" * 70)
    print(f"{'File':<30} {'Base(s)':>8} {'Now(s)':>8} {'Delta':>8} {'Speedup':>8}")
    print("-" * 70)

    for r in results:
        b = base_map.get(r["file"])
        if b is None:
            print(f"{r['file']:<30} {'(new)':>8} {r['total_time_s']:>8.1f}")
            continue
        delta = r["total_time_s"] - b["total_time_s"]
        speedup = b["total_time_s"] / r["total_time_s"] if r["total_time_s"] > 0 else 0
        sign = "+" if delta > 0 else ""
        print(f"{r['file']:<30} {b['total_time_s']:>8.1f} {r['total_time_s']:>8.1f} "
              f"{sign}{delta:>7.1f} {speedup:>7.2f}x")

    print("=" * 70 + "\n")


def main() -> None:
    """Main entry point for the benchmark runner."""
    parser = argparse.ArgumentParser(description="PDF-to-LaTeX Pipeline Benchmark")
    parser.add_argument("--input", type=str, default="data/test_samples",
                        help="Directory or single file to benchmark")
    parser.add_argument("--output", type=str, default="data/benchmark_output",
                        help="Output directory for benchmark results")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save results as baseline.json")
    parser.add_argument("--compare", type=str, default=None,
                        help="Path to baseline JSON to compare against")
    parser.add_argument("--mcts", action="store_true",
                        help="Enable MCTS for figure recognition")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect input files
    if input_path.is_file():
        files = [input_path]
    elif input_path.is_dir():
        supported = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
        files = sorted(
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported
        )
    else:
        print(f"Input not found: {input_path}")
        sys.exit(1)

    if not files:
        print(f"No supported files found in {input_path}")
        sys.exit(1)

    print(f"Benchmarking {len(files)} file(s) from {input_path}")
    print(f"Platform: {platform.system()} {platform.machine()}")

    pipeline = Pipeline(config)
    results: List[Dict[str, Any]] = []

    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f.name} ...", end=" ", flush=True)
        try:
            r = run_single(pipeline, f, output_dir, use_mcts=args.mcts)
            results.append(r)
            status = "OK" if r["success"] else "FAIL"
            print(f"{status} ({r['total_time_s']:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({
                "file": f.name, "success": False, "total_time_s": 0,
                "stage_times": {}, "ssim_score": None, "compilation_ok": False,
                "ram_delta_mb": 0, "ram_peak_mb": 0, "gpu_peak_mb": 0,
                "error": str(e),
            })

    print_summary(results)

    # Save results
    results_path = output_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    if args.save_baseline:
        baseline_path = output_dir / "baseline.json"
        with open(baseline_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Baseline saved to {baseline_path}")

    if args.compare:
        compare_baseline(results, Path(args.compare))


if __name__ == "__main__":
    main()
