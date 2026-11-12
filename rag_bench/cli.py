"""CLI entry point for rag-bench."""

import argparse
import ctypes
import os
import sys

from rag_bench.config import load_config
from rag_bench.pipeline import BenchmarkPipeline


def enforce_cpu_cap(percentage: float = 0.65) -> None:
    """Enforces that at most percentage (default 65-70%) of CPU cores are used."""
    cores = os.cpu_count() or 12
    limit = max(1, int(cores * percentage))
    os.environ["OMP_NUM_THREADS"] = str(limit)
    os.environ["MKL_NUM_THREADS"] = str(limit)
    os.environ["OPENBLAS_NUM_THREADS"] = str(limit)
    os.environ["NUMEXPR_NUM_THREADS"] = str(limit)
    try:
        import torch

        torch.set_num_threads(limit)
    except Exception:
        pass
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetPriorityClass.restype = ctypes.c_bool
            kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

            mask = (1 << limit) - 1
            h_process = kernel32.GetCurrentProcess()
            kernel32.SetProcessAffinityMask(h_process, mask)
            kernel32.SetPriorityClass(h_process, 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS
        except Exception:
            pass


def main() -> None:
    enforce_cpu_cap(0.65)
    parser = argparse.ArgumentParser(
        prog="rag-bench",
        description="Reproducible benchmark comparing retrieval pipelines on BEIR datasets",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. 'run' command
    run_parser = subparsers.add_parser("run", help="Run benchmark on a BEIR dataset")
    run_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["scifact", "nfcorpus", "fiqa"],
        help="Target BEIR dataset",
    )
    run_parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )

    # 2. 'report' command
    report_parser = subparsers.add_parser("report", help="Generate benchmark reports and plots")
    report_parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to results directory containing runs",
    )
    report_parser.add_argument(
        "--readme",
        type=str,
        default="README.md",
        help="Path to README.md to inject tables into",
    )

    # 3. 'serve' command
    serve_parser = subparsers.add_parser("serve", help="Run FastAPI serving layer and UI")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port")

    args = parser.parse_args()

    if args.command == "run":
        config = load_config(args.config)
        pipeline = BenchmarkPipeline(config)
        output_dir = pipeline.run_dataset(args.dataset)
        print(f"\nRun completed successfully. Results saved to: {output_dir}")

    elif args.command == "report":
        from rag_bench.report import generate_reports
        generate_reports(results_dir=args.results_dir, readme_path=args.readme)

    elif args.command == "serve":
        import uvicorn

        from rag_bench.serve import app
        print(f"Starting rag-bench server at http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
