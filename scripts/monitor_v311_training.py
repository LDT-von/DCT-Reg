#!/usr/bin/env python3
"""Real-time monitoring for v3.11 BLCA UNI training on GPU 1.

Usage:
    python monitor_v311_training.py
    python monitor_v311_training.py --interval 30

Shows:
- GPU 0/1 memory & utilization (with PID mapping)
- Per-fold progress (current epoch / max_epochs, last 5 batches)
- Estimated time remaining
- Recent training losses (per_slot_nll, diversity, IPCW rank)
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path("/data1/DCT-Reg")
LOG_DIR = REPO_ROOT / "logs" / "v311_blca_uni_fixed"
RESULTS_DIR = REPO_ROOT / "results" / "dct_v311_blca_uni_fixed" / "blca"


def get_gpu_stats() -> str:
    """Return formatted GPU memory and utilization."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True
        )
        rows = []
        for line in out.strip().split("\n"):
            idx, name, mem_used, mem_total, util = line.split(",")
            mem_pct = 100.0 * float(mem_used) / float(mem_total)
            rows.append(
                f"  GPU{idx} {name}: {mem_used:>6}MB / {mem_total}MB ({mem_pct:5.1f}%) "
                f"util={util}%"
            )
        return "\n".join(rows)
    except Exception as e:
        return f"  [nvidia-smi error: {e}]"


def get_running_processes() -> str:
    """Show GPU 0/1 process mappings."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
             "--format=csv,noheader,nounits"],
            text=True
        )
        if not out.strip():
            return "  (no GPU compute processes)"
        rows = []
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                uuid_short = parts[0].split("-")[0]
                rows.append(f"  uuid:{uuid_short}  PID {parts[1]:>8}  "
                            f"{parts[3]:>6}MB  {parts[2]}")
        return "\n".join(rows)
    except Exception as e:
        return f"  [nvidia-smi error: {e}]"


def parse_fold_progress(log_path: Path) -> dict:
    """Parse the most recent training progress from a fold log."""
    if not log_path.exists():
        return {"fold": log_path.stem, "found": False}

    with open(log_path, "r", errors="ignore") as f:
        content = f.read()

    fold = log_path.stem.replace("fold", "")
    info = {"fold": fold, "found": True, "current_epoch": 0, "total_epochs": 0,
            "batch": 0, "batches_total": 0, "loss": None, "surv": None,
            "last_loss": None, "eta": None, "elapsed_sec": 0,
            "completed": False}

    # Match current epoch progress (e.g., "[Fold 0] Epoch 12/30: 45%|▌▌ ")
    m = re.findall(r"Epoch (\d+)/(\d+):\s+(\d+)%", content)
    if m:
        epoch, total, pct = m[-1]
        info["current_epoch"] = int(epoch)
        info["total_epochs"] = int(total)

    # Match batch counter (e.g., "12/38 [01:23<...,")
    m = re.findall(r"(\d+)/(\d+)\s*\[([0-9:]+)<([^,]+),\s*([\d.]+)s/it", content)
    if m:
        b, bt, elapsed, eta, it_s = m[-1]
        info["batch"] = int(b)
        info["batches_total"] = int(bt)
        # elapsed like "01:23" -> 83 sec
        em = re.match(r"(\d+):(\d+)", elapsed)
        info["elapsed_sec"] = int(em.group(1)) * 60 + int(em.group(2)) if em else 0
        info["eta"] = eta.strip()

    # Match last loss
    losses = re.findall(r"batch[:=]?\s*(\d+)\s+loss:\s*([\d.]+)\s+surv:\s*([\d.]+)", content)
    if losses:
        _, loss, surv = losses[-1]
        info["loss"] = float(loss)
        info["surv"] = float(surv)

    # Check completion
    if "[best]" in content or "Final" in content or "results_final.pkl" in content:
        info["completed"] = True

    return info


def find_active_fold() -> tuple[int | None, Path | None]:
    """Detect which fold is currently running (most recent log mtime)."""
    if not LOG_DIR.exists():
        return None, None

    logs = sorted(LOG_DIR.glob("fold*.log"))
    if not logs:
        return None, None

    # Find the most recently updated log
    latest = max(logs, key=lambda p: p.stat().st_mtime)

    # If the latest log has been updated in the last 5 min, it's active
    age_sec = time.time() - latest.stat().st_mtime
    if age_sec > 300:  # 5 min idle
        return None, latest

    # Extract fold number from filename
    m = re.search(r"fold(\d+)", latest.name)
    if m:
        return int(m.group(1)), latest
    return None, latest


def find_master_pid() -> int | None:
    """Find the master launcher process ID."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "run_v311"], text=True)
        pids = [int(p) for p in out.strip().split("\n") if p]
        return pids[0] if pids else None
    except subprocess.CalledProcessError:
        return None


def find_python_pid(gpu_idx: int) -> int | None:
    """Find Python training PID on given GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
             "--format=csv,noheader,nounits"],
            text=True
        )
        # Parse to find gpu_idx via PCIe bus mapping
        uuid_out = subprocess.check_output(
            ["nvidia-smi", "-i", str(gpu_idx),
             "--query-gpu=uuid", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        for line in out.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == uuid_out:
                return int(parts[1])
    except Exception:
        pass
    return None


def render_progress(active_fold: int | None, fold_info: dict) -> str:
    """Render the per-fold progress line."""
    if active_fold is None:
        return "  No fold currently active"

    pct = 100.0 * fold_info["batch"] / max(1, fold_info["batches_total"])
    epoch_pct = 100.0 * fold_info["current_epoch"] / max(1, fold_info["total_epochs"])
    eta = fold_info.get("eta", "??:??")
    elapsed = fold_info.get("elapsed_sec", 0)
    elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"

    return (
        f"  Fold {active_fold}: "
        f"epoch {fold_info['current_epoch']}/{fold_info['total_epochs']} "
        f"({epoch_pct:5.1f}%) "
        f"batch {fold_info['batch']}/{fold_info['batches_total']} ({pct:5.1f}%) "
        f"loss={fold_info.get('loss', 'N/A')} "
        f"elapsed={elapsed_str} eta={eta}"
    )


def check_results_pkl() -> str:
    """Check which folds have completed (results_final.pkl exists)."""
    if not RESULTS_DIR.exists():
        return "  (results dir not yet created)"

    folds_done = []
    for fold in range(5):
        matches = list(RESULTS_DIR.rglob(f"split_{fold}_results_final.pkl"))
        if matches:
            # Get the most recent one
            latest = max(matches, key=lambda p: p.stat().st_mtime)
            age_min = (time.time() - latest.stat().st_mtime) / 60
            folds_done.append(f"fold{fold}: ✓ ({age_min:.0f}min ago)")

    if not folds_done:
        return "  (no folds completed yet)"
    return "  Completed: " + "  ".join(folds_done)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=15,
                        help="Refresh interval in seconds")
    parser.add_argument("--gpu", type=int, default=1, help="GPU to monitor")
    args = parser.parse_args()

    print("=" * 70)
    print(f" v3.11 BLCA UNI Training Monitor  (GPU {args.gpu})")
    print(f" Refresh: {args.interval}s   Press Ctrl+C to exit")
    print("=" * 70)

    while True:
        try:
            ts = time.strftime("%H:%M:%S")
            print(f"\n[{ts}]" + "─" * 60)

            # Section 1: GPUs
            print("GPU Status:")
            print(get_gpu_stats())
            print("Running processes:")
            print(get_running_processes())

            # Section 2: Master launcher
            master_pid = find_master_pid()
            print(f"\nLauncher:")
            print(f"  master script PID: {master_pid if master_pid else 'NOT FOUND'}")

            # Section 3: Per-fold progress
            print(f"\nActive fold:")
            active_fold, active_log = find_active_fold()
            if active_fold is not None and active_log is not None:
                info = parse_fold_progress(active_log)
                print(render_progress(active_fold, info))
                if info.get("completed"):
                    print("  ✓ fold completed!")
            else:
                if active_log:
                    print(f"  last activity: {active_log.name} "
                          f"({(time.time() - active_log.stat().st_mtime) / 60:.1f} min ago, idle)")
                else:
                    print("  (no logs found)")

            # Section 4: Completed folds
            print(f"\nCheckpoint status:")
            print(check_results_pkl())

            sys.stdout.flush()
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n[exit] monitor stopped")
            return 0
        except Exception as e:
            print(f"\n[error] {e}")
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
