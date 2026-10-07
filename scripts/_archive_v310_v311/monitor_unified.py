#!/usr/bin/env python3
"""Read-only monitor for DCT-Reg training queues.

Watches (no side effects):
  * GPU 0 memory / utilization via ``nvidia-smi``.
  * Per-GPU scheduler lock files under ``results/.locks/``.
  * Per-cancer task lock files under ``results/dct_v3.10/robust/final*/<cancer>/``.
  * Tail of the latest ``logs/run_dct_v310_*_queue.log`` file.
  * Last ``split_<N>_results_final.pkl`` per cancer (i.e. completed folds).

Use ``--watch`` to refresh every N seconds; default is a one-shot snapshot.

This script never starts, stops, or modifies any process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_ROOT = REPO_ROOT / "results" / ".locks"
QUEUE_LOG_GLOB = "run_dct_v310_*_queue.log"
FINAL_DIRS = [
    REPO_ROOT / "results" / "dct_v3.10" / "robust" / "final",
    REPO_ROOT / "results" / "dct_v3.10" / "robust" / "final_uni",
]


def gpu_snapshot() -> list[dict]:
    """Query nvidia-smi once.  Returns list of per-GPU dicts."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return [{"error": str(exc)}]
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 6:
            continue
        idx, name, util, mem_used, mem_total, temp = parts
        rows.append({
            "index": idx,
            "name": name,
            "util_pct": int(util),
            "mem_used_mib": int(mem_used),
            "mem_total_mib": int(mem_total),
            "temp_c": int(temp),
        })
    return rows


def lock_owner(path: Path) -> dict | None:
    """Read a lock JSON (pid, label)."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"raw": path.read_text(errors="replace")[:120]}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def scan_scheduler_locks() -> list[dict]:
    rows = []
    if not LOCK_ROOT.is_dir():
        return rows
    for lock in sorted(LOCK_ROOT.glob(".run_gpu_*.lock")) + \
                sorted(LOCK_ROOT.glob(".smoke_gpu_*.lock")):
        owner = lock_owner(lock)
        alive = bool(owner and "pid" in owner and _pid_alive(owner["pid"]))
        rows.append({
            "lock": lock.relative_to(REPO_ROOT).as_posix(),
            "alive": alive,
            "owner": owner,
        })
    return rows


def scan_task_locks(cancer_dirs: list[Path]) -> list[dict]:
    rows = []
    for d in cancer_dirs:
        if not d.is_dir():
            continue
        # Use rglob so we find locks either directly under <dir>/ or under <dir>/<cancer>/
        # (the queue places them at results/dct_v3.10/robust/final/<cancer>/.split_*.lock)
        for lock in sorted(d.rglob(".split_*.dct_reg.lock")):
            owner = lock_owner(lock)
            alive = bool(owner and "pid" in owner and _pid_alive(owner["pid"]))
            rows.append({
                "lock": lock.relative_to(REPO_ROOT).as_posix(),
                "alive": alive,
                "owner": owner,
            })
    return rows


def scan_completions(cancer_dirs: list[Path]) -> dict[str, list[int]]:
    """Per cancer, which folds have a final.pkl written."""
    out: dict[str, list[int]] = {}
    for d in cancer_dirs:
        if not d.is_dir():
            continue
        for cancer_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            cancer = cancer_dir.name
            folds = []
            for fk in sorted(cancer_dir.rglob("split_*_results_final.pkl")):
                # split_<N>_results_final.pkl
                stem = fk.name.split("_")[1]
                try:
                    folds.append(int(stem))
                except ValueError:
                    pass
            if folds:
                out[cancer] = sorted(set(folds))
    return out


def latest_queue_log() -> Path | None:
    logs_dir = REPO_ROOT / "logs"
    if not logs_dir.is_dir():
        return None
    candidates = sorted(logs_dir.glob(QUEUE_LOG_GLOB),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def tail(path: Path, n: int = 6) -> list[str]:
    if not path.is_file():
        return [f"(no log file: {path})"]
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - 4096), os.SEEK_SET)
        chunk = handle.read().decode("utf-8", errors="replace")
    return chunk.splitlines()[-n:]


def render(snap: dict) -> str:
    lines = []
    ts = snap["timestamp"]
    lines.append(f"=== DCT-Reg Queue Monitor @ {ts} ===")
    lines.append("")
    lines.append("[GPU]")
    if not snap["gpu"]:
        lines.append("  (no GPUs / nvidia-smi error)")
    for g in snap["gpu"]:
        if "error" in g:
            lines.append(f"  ERROR: {g['error']}")
            continue
        mem_pct = 100.0 * g["mem_used_mib"] / max(1, g["mem_total_mib"])
        lines.append(
            f"  GPU{g['index']} {g['name']}: util={g['util_pct']:3d}% "
            f"mem={g['mem_used_mib']:>6d}/{g['mem_total_mib']} MiB "
            f"({mem_pct:5.1f}%) temp={g['temp_c']}C"
        )
    lines.append("")
    lines.append("[Scheduler locks] results/.locks/.run_gpu_*.lock")
    if not snap["scheduler_locks"]:
        lines.append("  (none)")
    for r in snap["scheduler_locks"]:
        marker = "ALIVE" if r["alive"] else "STALE"
        owner = r["owner"]
        if owner and "pid" in owner:
            owner_str = f"pid={owner['pid']} label={owner.get('label','')}"
        else:
            owner_str = repr(owner)
        lines.append(f"  [{marker}] {r['lock']}  {owner_str}")
    lines.append("")
    lines.append("[Task locks] results/dct_v3.10/robust/final*/<cancer>/.split_*.lock")
    if not snap["task_locks"]:
        lines.append("  (none)")
    for r in snap["task_locks"]:
        marker = "ALIVE" if r["alive"] else "STALE"
        owner = r["owner"]
        if owner and "pid" in owner:
            owner_str = f"pid={owner['pid']} label={owner.get('label','')}"
        else:
            owner_str = repr(owner)
        lines.append(f"  [{marker}] {r['lock']}  {owner_str}")
    lines.append("")
    lines.append("[Completed folds] split_<N>_results_final.pkl")
    if not snap["completions"]:
        lines.append("  (no completions yet)")
    for cancer, folds in snap["completions"].items():
        lines.append(f"  {cancer:<10} folds: {folds}")
    lines.append("")
    lines.append(f"[Latest queue log] {snap['latest_log']}")
    for ln in snap["log_tail"]:
        lines.append(f"  | {ln}")
    lines.append("")
    return "\n".join(lines)


def take_snapshot() -> dict:
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": gpu_snapshot(),
        "scheduler_locks": scan_scheduler_locks(),
        "task_locks": scan_task_locks(FINAL_DIRS),
        "completions": scan_completions(FINAL_DIRS),
        "latest_log": latest_queue_log(),
        "log_tail": tail(latest_queue_log()) if latest_queue_log() else [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", type=float, default=0.0,
                    help="If >0, refresh every N seconds instead of one-shot.")
    ap.add_argument("--tail-lines", type=int, default=6)
    args = ap.parse_args()
    if args.watch <= 0:
        print(render(take_snapshot()))
        return 0
    try:
        while True:
            os.system("clear")
            print(render(take_snapshot()))
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
