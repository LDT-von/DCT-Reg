#!/usr/bin/env python3
"""Monitor v3.11 fixed v2 training progress.

Reads logs/v311_blca_uni_fixed_v2/fold*.log and prints per-modality variance
trends. Run periodically to track progress.

Usage:
    python scripts/monitor_v311_v2_training.py
    python scripts/monitor_v311_v2_training.py --watch 60  # refresh every 60s
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

REPO = Path("/data1/DCT-Reg")
LOG_DIR = REPO / "logs/v311_blca_uni_fixed_v2"

VAR_MIN, VAR_MAX = 0.005, 0.05


def parse_fold(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    text = log_path.read_text(errors="ignore")
    rows = []
    for line in text.split("\n"):
        if "[Epoch" not in line or "train_loss=" not in line:
            continue
        m = re.search(r"\[Epoch (\d+)\]", line)
        if not m:
            continue
        row = {"epoch": int(m.group(1))}
        for f in [
            "v311_per_slot_nll", "v311_slot_diversity",
            "v311_slot_variance", "v311_slot_variance_wsi",
            "v311_slot_variance_omic", "anchor_coverage",
        ]:
            m2 = re.search(rf"{f}=([\d.\-e]+)", line)
            row[f] = float(m2.group(1)) if m2 else None
        # Val cindex from next line
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=int, default=0,
                        help="If >0, refresh every N seconds")
    args = parser.parse_args()

    while True:
        print("\033[2J\033[H", end="")  # clear screen
        print("=" * 100)
        print(" v3.11 Fixed v2 — Training Monitor")
        print("=" * 100)

        any_fold = False
        for fold in range(5):
            rows = parse_fold(LOG_DIR / f"fold{fold}.log")
            if not rows:
                if fold == 0:
                    print(f"Fold {fold}: (no data yet)")
                continue
            any_fold = True
            last = rows[-1]
            in_band_wsi = (last.get("v311_slot_variance_wsi") is not None
                           and VAR_MIN <= last["v311_slot_variance_wsi"] <= VAR_MAX)
            in_band_omic = (last.get("v311_slot_variance_omic") is not None
                            and VAR_MIN <= last["v311_slot_variance_omic"] <= VAR_MAX)
            cov = last.get("anchor_coverage")
            nll = last.get("v311_per_slot_nll")
            div = last.get("v311_slot_diversity")

            wsi_str = f"{last.get('v311_slot_variance_wsi', 0):.5f}"
            omic_str = f"{last.get('v311_slot_variance_omic', 0):.5f}"
            nll_str = f"{nll:.4f}" if nll else "N/A"
            div_str = f"{div:.4f}" if div else "N/A"
            cov_str = f"{cov:.3f}" if cov else "N/A"

            print(f"\nFold {fold}  (epoch {last['epoch']})")
            print(f"  var_wsi  = {wsi_str}  {'✅' if in_band_wsi else '❌'}")
            print(f"  var_omic = {omic_str}  {'✅' if in_band_omic else '❌'}")
            print(f"  per_slot_nll = {nll_str}  diversity = {div_str}")
            print(f"  anchor_coverage = {cov_str}")

            # Check if variance is rising
            if len(rows) >= 5:
                recent_wsi = [r.get("v311_slot_variance_wsi") for r in rows[-5:]
                              if r.get("v311_slot_variance_wsi") is not None]
                recent_omic = [r.get("v311_slot_variance_omic") for r in rows[-5:]
                               if r.get("v311_slot_variance_omic") is not None]
                if recent_wsi:
                    print(f"  Recent WSI variance trend: {' → '.join(f'{v:.4f}' for v in recent_wsi)}")
                if recent_omic:
                    print(f"  Recent Omic variance trend: {' → '.join(f'{v:.4f}' for v in recent_omic)}")

        if not any_fold:
            print("\n(No fold logs found yet — training may be initializing)")

        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
