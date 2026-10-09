#!/usr/bin/env python3
"""Real-time score monitoring for v3.11 BLCA UNI 5-fold training.

Usage:
    python monitor_scores.py
    python monitor_scores.py --interval 30
    python monitor_scores.py --fold 0        # only fold 0
    python monitor_scores.py --recent 5       # show last N epochs

Shows per-fold training curves:
  - val_cindex (primary), val_ipcw, val_IBS, val_iauc
  - train_loss, train_cindex
  - v311_per_slot_nll, v311_slot_diversity, v311_slot_variance
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path("/data1/DCT-Reg")
LOG_DIR = REPO_ROOT / "logs" / "v311_blca_uni_fixed"
NUM_FOLDS = 5


# ----------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------

def parse_epoch_metrics(content: str) -> dict:
    """Parse all epoch metrics from full log content (handles multi-line format).

    Training metrics: '[Epoch N] train_loss=X ...'
    Val metrics on SAME line: '[Epoch N] val cindex=X ipcw=X IBS=X iauc=X'
    Val metrics on NEXT line: '  [Epoch N] val cindex=X ...'
    """
    result = {}

    # 1. Extract epoch number from [Epoch N] anywhere
    m = re.search(r"\[Epoch (\d+)\]", content)
    if m:
        result["epoch"] = int(m.group(1))

    # 2. Train metrics
    for key in ("train_loss", "train_cindex", "ot", "ipcw_rank",
                "v311_per_slot_nll", "v311_slot_diversity", "v311_slot_variance",
                "v311_per_slot_nll_lambda", "v311_slot_diversity_lambda"):
        m2 = re.search(rf"{key}=([\d.e+-]+)", content)
        if m2:
            try:
                result[key] = float(m2.group(1))
            except ValueError:
                pass

    # 3. Val metrics: "val cindex=X ipcw=X IBS=X iauc=X"
    m3 = re.search(
        r"val cindex=([\d.]+)\s+ipcw=([\d.]+)\s+IBS=([\d.]+)\s+iauc=([\d.]+)",
        content,
    )
    if m3:
        result.update({
            "val_cindex": float(m3.group(1)),
            "val_ipcw": float(m3.group(2)),
            "val_IBS": float(m3.group(3)),
            "val_iauc": float(m3.group(4)),
        })

    return result


def parse_fold_scores(log_path: Path) -> list[dict]:
    """Read all epoch metrics from a fold log.

    Each epoch produces:
      [Epoch N] train_loss=... v311_per_slot_nll=... v311_slot_diversity=...
      [Epoch N] val cindex=... ipcw=... IBS=... iauc=...

    We accumulate into a dict indexed by epoch number so that the train and
    val lines for the same epoch (which come from two consecutive log lines)
    merge into one record.
    """
    if not log_path.exists():
        return []

    with open(log_path, errors="ignore") as f:
        content = f.read()

    epoch_metric_re = re.compile(r"\[Epoch (\d+)\]")

    # Parse line by line (val often follows train on next line, but both have [Epoch N])
    by_epoch: dict[int, dict] = {}
    for line in content.split("\n"):
        m = epoch_metric_re.search(line)
        if not m:
            continue
        ep = int(m.group(1))
        record = by_epoch.setdefault(ep, {"epoch": ep})

        # Try parsing as train-metrics line (has train_loss=)
        if "train_loss=" in line:
            for key in ("train_loss", "train_cindex", "ot", "ipcw_rank",
                        "v311_per_slot_nll", "v311_slot_diversity",
                        "v311_slot_variance",
                        "v311_slot_variance_wsi",
                        "v311_slot_variance_omic",
                        "v311_per_slot_nll_lambda",
                        "v311_slot_diversity_lambda"):
                m2 = re.search(rf"{key}=([\d.e+-]+)", line)
                if m2:
                    try:
                        record[key] = float(m2.group(1))
                    except ValueError:
                        pass

        # Try parsing as val-metrics line (has val cindex=)
        if "val cindex=" in line:
            m3 = re.search(
                r"val cindex=([\d.]+)\s+ipcw=([\d.]+)\s+IBS=([\d.]+)\s+iauc=([\d.]+)",
                line,
            )
            if m3:
                record["val_cindex"] = float(m3.group(1))
                record["val_ipcw"] = float(m3.group(2))
                record["val_IBS"] = float(m3.group(3))
                record["val_iauc"] = float(m3.group(4))

    return [by_epoch[k] for k in sorted(by_epoch.keys())]


def get_active_epoch(log_path: Path) -> int:
    """Get the currently training epoch from a fold log."""
    if not log_path.exists():
        return 0
    with open(log_path, errors="ignore") as f:
        content = f.read()
    epochs = re.findall(r"Epoch (\d+)/(\d+):", content)
    if epochs:
        return int(epochs[-1][0])
    return 0


def get_active_batch(log_path: Path) -> tuple[int, int]:
    """Get current batch progress from log (last (cur,total) pair)."""
    if not log_path.exists():
        return 0, 0
    with open(log_path, errors="ignore") as f:
        content = f.read()
    m = re.findall(r"Epoch \d+/\d+:\s+\d+%\|.*?(\d+)/(\d+)\s*\[", content)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    return 0, 0
    """Get current batch progress."""
    if not log_path.exists():
        return 0, 0
    with open(log_path, errors="ignore") as f:
        content = f.read()
    m = re.findall(r"Epoch \d+/\d+:\s+\d+%\|.*?(\d+)/(\d+)\s*\[", content)
    if m:
        return int(m[-1][0]), int(m[-1][1])
    return 0, 0


# ----------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------

def fmt(val: float, decimals: int = 4) -> str:
    if val is None:
        return "  ----"
    return f"{val:>{decimals+4}.{decimals}f}"


def best_epoch(scores: list[dict], key: str = "val_cindex") -> tuple[int, float]:
    """Return (best_epoch, best_value) for a metric."""
    if not scores:
        return 0, 0.0
    best_e, best_v = 0, -999.0
    for s in scores:
        v = s.get(key, -999.0)
        if v is not None and v > best_v:
            best_v = v
            best_e = s["epoch"]
    return best_e, best_v


def render_fold_table(fold_scores: list[dict], fold: int,
                      active_epoch: int, n_recent: int) -> str:
    """Render one fold as an ASCII table."""
    n_total_epochs = max((s["epoch"] for s in fold_scores), default=0) + 1
    show_epochs = list(range(max(0, n_total_epochs - n_recent), n_total_epochs))
    if not show_epochs:
        return f"Fold {fold}: (no scores yet)"

    # header
    header = (
        f"Fold {fold}  ({len(fold_scores)} epochs, active epoch {active_epoch})\n"
        f"  {'ep':>3} | {'train_loss':>10} | {'train_c':>8} | "
        f"{'v311_psnll':>10} | {'v311_div':>9} | "
        f"{'v311_varW':>9} | {'v311_varO':>9} | "
        f"{'val_c':>7} | {'val_ipcw':>9} | {'val_IBS':>8} | {'val_iauc':>8} | "
        f"{'★best_c':>8}"
    )
    lines = [header]

    for ep in show_epochs:
        s = next((x for x in fold_scores if x["epoch"] == ep), None)
        if s is None:
            lines.append(f"  {ep:>3} | (no data)")
            continue

        best_e, best_v = best_epoch(fold_scores[:ep + 1], "val_cindex")
        is_best = " ◀" if s["epoch"] == best_e else ""

        lines.append(
            f"  {s['epoch']:>3} | "
            f"{fmt(s.get('train_loss'))} | "
            f"{fmt(s.get('train_cindex'), 4)} | "
            f"{fmt(s.get('v311_per_slot_nll'), 4)} | "
            f"{fmt(s.get('v311_slot_diversity'), 4)} | "
            f"{fmt(s.get('v311_slot_variance_wsi') or s.get('v311_slot_variance'), 4)} | "
            f"{fmt(s.get('v311_slot_variance_omic') or s.get('v311_slot_variance'), 4)} | "
            f"{fmt(s.get('val_cindex'), 4)} | "
            f"{fmt(s.get('val_ipcw'), 4)} | "
            f"{fmt(s.get('val_IBS'), 4)} | "
            f"{fmt(s.get('val_iauc'), 4)} | "
            f"{best_v:>8.4f}{is_best}"
        )
    return "\n".join(lines)


def render_summary(fold_scores: list[list[dict]]) -> str:
    """Show best val_cindex per fold."""
    rows = ["", "─" * 70, "Best val_cindex per fold:"]
    for fold, scores in enumerate(fold_scores):
        if not scores:
            rows.append(f"  Fold {fold}: (in progress / no data yet)")
            continue
        best_ep, best_v = best_epoch(scores, "val_cindex")
        best_ipcw_ep, best_ipcw = best_epoch(scores, "val_ipcw")
        avg_so_far = sum(s.get("val_cindex", 0) for s in scores) / len(scores)
        rows.append(
            f"  Fold {fold}: best={best_v:.4f} @ep{best_ep}  "
            f"ipcw={best_ipcw:.4f} @ep{best_ipcw_ep}  "
            f"avg_last5={avg_so_far:.4f}"
        )
    # overall
    completed = [best_epoch(scores, "val_cindex")[1] for scores in fold_scores if scores]
    if completed:
        rows.append(f"  Overall mean: {sum(completed)/len(completed):.4f}")
    return "\n".join(rows)


def get_gpu_status() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0 or not out.stdout:
            # Fall back to file-based query
            return "(GPU status unavailable)"
        lines = []
        for line in out.stdout.strip().split("\n"):
            parts = line.split(",")
            if len(parts) >= 3:
                idx, mem, util = parts[0], parts[1], parts[2]
                try:
                    pct = 100 * float(mem.strip()) / 32607
                    lines.append(f"GPU{idx}: {mem.strip()}MB ({pct:.0f}%) util={util.strip()}%")
                except ValueError:
                    pass
        return "  ".join(lines) if lines else "(GPU status unavailable)"
    except Exception:
        return "(GPU status unavailable)"


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=15,
                        help="Refresh interval (seconds)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Monitor only one fold (0-4)")
    parser.add_argument("--recent", type=int, default=8,
                        help="Show last N epochs per fold")
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Directory with fold{N}.log files (overrides default LOG_DIR)")
    parser.add_argument("--log-file", type=str, default=None,
                        help="Single combined log file containing all folds (will be sliced by '[Fold K]' markers)")
    parser.add_argument("--label", type=str, default=None,
                        help="Title shown in monitor header")
    args = parser.parse_args()

    global LOG_DIR
    if args.log_dir:
        LOG_DIR = Path(args.log_dir)
    log_files = {}  # fold -> Path
    if args.log_file:
        # Single-file mode: synthesize fold logs by slicing on '[Fold K]' markers
        single = Path(args.log_file)
        slices_dir = LOG_DIR / f"_slices_{single.stem}"
        slices_dir.mkdir(parents=True, exist_ok=True)
        try:
            content = single.read_text(errors="ignore")
        except Exception:
            content = ""
        import re as _re
        pat = _re.compile(r"\[Fold (\d+)\]")
        # Walk through line by line and split into fold-specific files
        current = None
        out_handles = {}
        slices = {}
        for line in content.splitlines(keepends=True):
            m = pat.search(line)
            if m:
                k = int(m.group(1))
                if k in out_handles:
                    out_handles[k].close()
                fp = slices_dir / f"fold{k}.log"
                slices.setdefault(k, fp)
                out_handles[k] = open(fp, "w")
                current = k
            if current is not None and current in out_handles:
                out_handles[current].write(line)
        for h in out_handles.values():
            h.close()
        log_files = slices
    folds_to_watch = ([args.fold] if args.fold is not None else
                      sorted(log_files.keys()) if log_files else list(range(NUM_FOLDS)))

    print("=" * 72)
    label = args.label or "v3.11 BLCA UNI"
    print(f" {label} — Score Monitor")
    print(f" Watching folds: {folds_to_watch}  |  Recent {args.recent} epochs")
    print(f" LOG_DIR: {LOG_DIR}")
    print("=" * 72)

    def get_scores_for(fold: int) -> list[dict]:
        if log_files:
            return parse_fold_scores(log_files[fold])
        return parse_fold_scores(LOG_DIR / f"fold{fold}.log")
    def get_epoch_for(fold: int) -> int:
        if log_files:
            return get_active_epoch(log_files[fold])
        return get_active_epoch(LOG_DIR / f"fold{fold}.log")
    def get_batch_for(fold: int) -> tuple[int, int]:
        if log_files:
            return get_active_batch(log_files[fold])
        return get_active_batch(LOG_DIR / f"fold{fold}.log")

    while True:
        try:
            ts = time.strftime("%H:%M:%S")
            print(f"\n╔═══ [{ts}] ════════════════════════════════════════════════════╗")

            # GPU status
            print(f"║ GPU: {get_gpu_status()}")
            print(f"╠════════════════════════════════════════════════════════════════╣")

            fold_all_scores: list[list[dict]] = []

            for fold in folds_to_watch:
                scores = get_scores_for(fold)
                fold_all_scores.append(scores)
                active_ep = get_epoch_for(fold)
                batch, total_b = get_batch_for(fold)
                n_done = len(scores)
                total_epochs_printed = active_ep + 1
                is_complete = n_done >= max(1, total_epochs_printed - 1) and active_ep >= 29
                if active_ep > n_done:
                    status = "TRAINING"
                elif is_complete:
                    status = "DONE"
                else:
                    status = "EVAL"
                print(f"╠══ Fold {fold} [{status:>7s}] ════════════════════════════════════════╣")
                print(f"║  epochs done: {n_done}/30   current: epoch {active_ep}  batch {batch}/{total_b}")
                if scores:
                    print(render_fold_table(scores, fold, active_ep, args.recent))
                else:
                    print("║  (waiting for first epoch to complete...)")

            # Summary
            print(render_summary(fold_all_scores))
            print("╚════════════════════════════════════════════════════════════════╝")

            sys.stdout.flush()
            time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\n[exit] monitor stopped")
            return 0
        except Exception as e:
            print(f"\n[error] {e}")
            import traceback
            traceback.print_exc()
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
