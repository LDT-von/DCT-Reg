#!/usr/bin/env python3
"""Monitor the v3.11 BLCA per-slot hazard export job.

Watches three independent signals every N seconds (default 30):
  1. **Process status**: is the export Python process still alive?
  2. **Log tail**: last 5 lines of /tmp/export_fold0.log, log size, mtime.
  3. **Output dir**: which artifacts (`per_slot_hazard.csv` / `.pkl` / `summary.txt`)
     have appeared so far on disk.

Usage:
    python scripts/monitor_export.py                # monitor default job
    python scripts/monitor_export.py --interval 10  # poll every 10s
    python scripts/monitor_export.py --log /tmp/foo.log --out_dir /tmp/foo
    python scripts/monitor_export.py --once          # print a single snapshot, then exit

Author: Cursor assistant for /data1/DCT-Reg, 2026-09-10.
"""
import argparse
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import psutil  # pip install psutil; if missing, falls back to subprocess
except ImportError:
    psutil = None


DEFAULT_PATTERN = "export_v311_per_slot_hazard"


def find_pid(match_substr):
    """Find PID whose cmdline matches substring; returns pid or None."""
    if psutil is not None:
        try:
            for proc in psutil.process_iter(["pid", "cmdline"]):
                try:
                    cmdline = proc.info.get("cmdline") or []
                    full = " ".join(cmdline)
                    if match_substr in full:
                        return proc.info["pid"], full
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            return None, None
        except Exception as e:
            # If psutil itself fails for any reason, fall through to pgrep.
            print(f"[monitor] psutil failed ({e!r}), falling back to pgrep",
                  file=sys.stderr)
    # Fallback: use pgrep
    import subprocess
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", match_substr], text=True
        ).strip().splitlines()
        if out:
            pid = int(out[0])
            cmd = subprocess.check_output(
                ["ps", "-p", str(pid), "-o", "args="], text=True
            ).strip()
            return pid, cmd
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None, None


def file_info(path):
    """Return (exists, size_bytes, mtime_str) for a path, or (-, -, 'missing')."""
    p = Path(path)
    if not p.exists():
        return False, -1, "missing"
    try:
        stat = p.stat()
        return True, stat.st_size, datetime.fromtimestamp(stat.st_mtime).strftime("%H:%M:%S")
    except OSError as e:
        return False, -1, f"err({e})"


def format_size(n_bytes):
    """Human-readable file size."""
    if n_bytes is None or n_bytes < 0:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def tail_lines(path, n=5):
    """Last N lines of a file (efficient tail)."""
    p = Path(path)
    if not p.exists():
        return [f"<no log file at {path}>"]
    try:
        # Use shell tail (efficient on large files). Fallback to naive read.
        import subprocess
        return subprocess.check_output(
            ["tail", "-n", str(n), str(p)], text=True
        ).splitlines()
    except Exception:
        try:
            with p.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 65536), os.SEEK_SET)
                lines = f.read().decode(errors="replace").splitlines()
            return lines[-n:]
        except Exception as e:
            return [f"<log read failed: {e}>"]


def print_snapshot(args, prev_log_size=0):
    """Print a one-shot monitor snapshot."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[monitor]  {now}")
    print(f"{'='*70}")

    # Signal 1: process status
    print("\n[1] PROCESS")
    pid, full = find_pid(args.match)
    if pid is None:
        print(f"  ❌  No process matching '{args.match}' is running.")
    else:
        elapsed_s = 0
        try:
            if psutil is not None:
                elapsed_s = time.time() - psutil.Process(pid).create_time()
        except Exception:
            pass
        e_min = int(elapsed_s // 60)
        e_sec = int(elapsed_s % 60)
        # Truncate command for display
        cmd_disp = full if len(full) < 90 else full[:87] + "..."
        print(f"  ✅  PID={pid}  alive={e_min}m{e_sec}s")
        print(f"       {cmd_disp}")

    # Signal 2: log status
    print(f"\n[2] LOG: {args.log}")
    exists, size, mtime = file_info(args.log)
    if exists:
        delta_s = ""
        if prev_log_size and size != prev_log_size:
            delta_s = f"  (Δ since last check: +{size - prev_log_size} bytes)"
        print(f"  ✅  size={format_size(size)}  mtime={mtime}{delta_s}")
        print("  tail:")
        for line in tail_lines(args.log, n=5):
            print(f"    {line}")
    else:
        print(f"  ❌  {args.log} does not exist yet.")

    # Signal 3: output dir status
    print(f"\n[3] OUTPUT DIR: {args.out_dir}")
    out = Path(args.out_dir)
    if not out.exists():
        print("  ❌  dir does not exist yet")
    else:
        # List relevant artifacts
        for fname in ("per_slot_hazard.csv", "per_slot_hazard.pkl", "summary.txt"):
            fpath = out / fname
            exists, size, mtime = file_info(fpath)
            status = f"size={format_size(size):>8}" if exists else "   missing"
            icon = "📄" if exists else "  "
            print(f"  {icon} {fname:<24} {status}")
        # Free disk on parent
        try:
            parent = out.parent if out.parent.exists() else out
            free = shutil.disk_usage(str(parent)).free
            print(f"  💽 free disk: {format_size(free)} on {parent}")
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Monitor the v3.11 export job.")
    parser.add_argument("--log", type=str,
                        default="/tmp/export_fold0.log",
                        help="Path to the log file being tee'd by the export script.")
    parser.add_argument("--out_dir", type=str,
                        default="/data1/DCT-Reg/results/dct_v311_blca_uni2h/per_slot_export",
                        help="Output dir where artifacts will appear.")
    parser.add_argument("--match", type=str,
                        default=DEFAULT_PATTERN,
                        help="Process cmdline substring to match (default: script name).")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between checks (default 30).")
    parser.add_argument("--once", action="store_true",
                        help="Print a single snapshot and exit.")
    args = parser.parse_args()

    prev_log_size = 0
    while True:
        try:
            _, cur_size, _ = file_info(args.log)
            print_snapshot(args, prev_log_size=cur_size if cur_size > 0 else prev_log_size)
            prev_log_size = cur_size if cur_size > 0 else prev_log_size
        except KeyboardInterrupt:
            print("\n[monitor] user interrupted, exiting.")
            return 0
        if args.once:
            return 0
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[monitor] user interrupted, exiting.")
            return 0


if __name__ == "__main__":
    main()
