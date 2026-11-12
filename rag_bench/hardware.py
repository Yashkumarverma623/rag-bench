"""Hardware, environment, and run manifest collection."""

import ctypes
import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

import torch


def get_git_sha() -> str:
    """Get current git commit hash."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def get_file_sha256(path: Path | str) -> str:
    """Compute sha256 of a file if it exists."""
    p = Path(path)
    if not p.exists():
        return "missing"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def get_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        if platform.system() == "Windows":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return float(round(stat.ullTotalPhys / (1024**3), 2))
        elif hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_PHYS_PAGES")  # type: ignore[attr-defined]
                page_size = os.sysconf("SC_PAGE_SIZE")  # type: ignore[attr-defined]
                return float(round((pages * page_size) / (1024**3), 2))
            except Exception:
                pass
    except Exception:
        pass
    return 0.0


def collect_manifest(
    dataset_name: str,
    dataset_checksums: Dict[str, str],
    seed: int = 42,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Collect complete hardware and environment manifest for reproducibility."""
    lockfile_path = Path("uv.lock")
    lockfile_hash = get_file_sha256(lockfile_path) if lockfile_path.exists() else "none"

    manifest: Dict[str, Any] = {
        "dataset": dataset_name,
        "seed": seed,
        "git_sha": get_git_sha(),
        "lockfile_sha256": lockfile_hash,
        "dataset_checksums": dataset_checksums,
        "hardware": {
            "os": f"{platform.system()} {platform.release()}",
            "os_version": platform.version(),
            "cpu_model": platform.processor() or "AMD64/x86_64",
            "logical_cores": os.cpu_count() or 1,
            "ram_gb": get_ram_gb(),
            "gpu_used": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "torch_thread_count": torch.get_num_threads(),
        },
        "models": {
            "dense_model": "BAAI/bge-small-en-v1.5",
            "dense_revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_revision": "233902d25c440f23af6f7d6e94d2946bac0bee0a",
        },
    }

    if extra_metadata:
        manifest["extra"] = extra_metadata

    return manifest
