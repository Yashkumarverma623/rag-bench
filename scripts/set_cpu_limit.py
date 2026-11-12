"""Utility to limit CPU usage of all Python processes to 80% or below capacity."""

import ctypes
import os
import subprocess
import sys


def limit_cpu_usage(max_fraction: float = 0.70) -> None:
    cores = os.cpu_count() or 12
    # Ensure strictly <= 70% of cores so total CPU never exceeds 80% with system overhead
    allowed_cores = max(1, int(cores * max_fraction))
    mask = (1 << allowed_cores) - 1

    print(f"Total CPU Cores: {cores}")
    print(f"Applying Hard Kernel Affinity Mask: {allowed_cores}/{cores} cores ({allowed_cores/cores*100:.1f}% CPU capacity)")
    print(f"Affinity Mask: 0x{mask:X} ({bin(mask)})")

    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        PROCESS_SET_INFORMATION = 0x0200
        PROCESS_QUERY_INFORMATION = 0x0400

        # Proper 64-bit Windows API prototypes
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_bool, ctypes.c_uint32]

        kernel32.SetProcessAffinityMask.restype = ctypes.c_bool
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]

        kernel32.SetPriorityClass.restype = ctypes.c_bool
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000

        try:
            out = subprocess.check_output(
                ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
                text=True,
            )
            for line in out.strip().splitlines()[1:]:
                parts = [p.strip('"') for p in line.split(",")]
                if len(parts) >= 2 and parts[1].isdigit():
                    pid = int(parts[1])
                    h = kernel32.OpenProcess(
                        PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION, False, pid
                    )
                    if h:
                        aff_ok = kernel32.SetProcessAffinityMask(h, mask)
                        prio_ok = kernel32.SetPriorityClass(h, BELOW_NORMAL_PRIORITY_CLASS)
                        kernel32.CloseHandle(h)
                        print(f"  PID {pid:6d}: ProcessorAffinity set -> {aff_ok}, Priority BelowNormal -> {prio_ok}")
        except Exception as e:
            print(f"Error updating processes: {e}")


if __name__ == "__main__":
    limit_cpu_usage(0.70)
