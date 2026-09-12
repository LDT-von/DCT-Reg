#!/usr/bin/env python3
"""
实时监控 v3.11 fixed v2 训练：
- 每 30 秒检查一次
- Fold 完成 / variance 进入目标区间时输出 ALERT
- 训练结束后输出汇总
"""
import os
import re
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("/data1/DCT-Reg/logs/v311_blca_uni_fixed_v2")
RESULT_DIR = Path("/data1/DCT-Reg/results/dct_v311_blca_uni_fixed_v2/blca/blca/SurvOTRank_dct_v311_slot_interpretable/0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_v2")
STATE_FILE = Path("/data1/DCT-Reg/.monitor_v311_v2.json")
TARGET_VARIANCE_MIN = 0.001
TARGET_VARIANCE_MAX = 0.05

last_seen_epoch = {0: -1, 1: -1, 2: -1, 3: -1, 4: -1}
completed_folds = set()
alert_history = []

def get_gpu_status():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
                                       "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL).decode()
        gpus = []
        for line in out.strip().split("\n"):
            idx, util, mem = line.split(", ")
            gpus.append({"idx": int(idx), "util": int(util), "mem": int(mem)})
        return gpus
    except Exception as e:
        return [{"err": str(e)}]

def get_epoch_curve(fold):
    csv = RESULT_DIR / f"epoch_curve_fold{fold}.csv"
    if not csv.exists():
        return None
    rows = []
    with open(csv) as f:
        header = None
        for line in f:
            parts = line.rstrip().split(",")
            if header is None:
                header = parts
                continue
            try:
                ep = int(parts[0])
                val_c = float(parts[1]) if parts[1] != "nan" else None
                psnll = float(parts[8]) if len(parts) > 8 and parts[8] != "nan" else None
                div = float(parts[9]) if len(parts) > 9 and parts[9] != "nan" else None
                varW = float(parts[11]) if len(parts) > 11 and parts[11] != "nan" else None
                varO = float(parts[12]) if len(parts) > 12 and parts[12] != "nan" else None
                rows.append({"ep": ep, "val_c": val_c, "psnll": psnll, "div": div, "varW": varW, "varO": varO})
            except (ValueError, IndexError):
                continue
    return rows

def get_progress(fold):
    log = LOG_DIR / f"fold{fold}.log"
    if not log.exists():
        return None
    try:
        text = log.read_text(errors="ignore")
        # Find last "Epoch X/30: ...batch=Y/38"
        matches = re.findall(r"Epoch (\d+)/30:\s+(\d+)%\|.*?\| (\d+)/(\d+)", text)
        if not matches:
            return None
        ep, pct, batch, total = matches[-1]
        return {"epoch": int(ep), "batch": int(batch), "total_batches": int(total)}
    except Exception:
        return None

def get_best_cindex(fold):
    log = LOG_DIR / f"fold{fold}.log"
    if not log.exists():
        return None
    text = log.read_text(errors="ignore")
    matches = re.findall(r"best cindex=([\d.]+)\s+@epoch\s+(\d+)", text)
    if not matches:
        return None
    c, ep = matches[-1]
    return {"cindex": float(c), "epoch": int(ep)}

def get_progress_bar(fold):
    prog = get_progress(fold)
    if not prog:
        return ""
    ep, batch, total = prog["epoch"], prog["batch"], prog["total_batches"]
    pct_epoch = (ep - 1) / 30 + batch / total / 30
    bar_len = 30
    filled = int(bar_len * pct_epoch)
    bar = "█" * filled + "░" * (bar_len - filled)
    return f"[Fold {fold}] {bar} {pct_epoch*100:5.1f}%  ep={ep}/30 batch={batch}/{total}"

def check_variance_alert(curve):
    """Detect WSI/Omics variance entering [0.001, 0.05] range for the first time"""
    alerts = []
    for row in curve:
        for slot, name in [("varW", "WSI"), ("varO", "Omics")]:
            v = row.get(slot)
            if v is None:
                continue
            alert_key = f"ep{row['ep']}_{slot}_in_range"
            if TARGET_VARIANCE_MIN <= v <= TARGET_VARIANCE_MAX and alert_key not in alert_history:
                alerts.append(f"  ✓ {name} var={v:.6f} ENTERED [{TARGET_VARIANCE_MIN},{TARGET_VARIANCE_MAX}] @ ep {row['ep']}")
                alert_history.append(alert_key)
    return alerts

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitor v3.11 fixed v2 started (GPU 0 only)")
    print("=" * 80)

    last_summary_time = 0
    while True:
        now = time.time()
        ts = datetime.now().strftime('%H:%M:%S')

        # Check training process alive
        try:
            ps_out = subprocess.check_output(["pgrep", "-f", "cli train"], stderr=subprocess.DEVNULL).decode().strip()
            procs = [int(x) for x in ps_out.split("\n") if x]
        except Exception:
            procs = []

        # Per-fold status
        any_active = False
        fold_summary = {}
        for fold in range(5):
            log_file = LOG_DIR / f"fold{fold}.log"
            best = get_best_cindex(fold)
            prog = get_progress(fold)
            curve = get_epoch_curve(fold)
            fold_summary[fold] = {"best": best, "progress": prog, "curve_len": len(curve) if curve else 0}

            # Check if fold just completed
            if best and fold not in completed_folds and curve:
                last_ep = curve[-1]["ep"] if curve else 0
                if last_ep >= 29 and fold not in completed_folds:
                    completed_folds.add(fold)
                    print(f"\n[{ts}] 🏁 ** FOLD {fold} COMPLETED ** best c={best['cindex']:.4f} @ ep {best['epoch']}")
                    print(f"  Final variance: WSI={curve[-1].get('varW', 0):.6f}, Omics={curve[-1].get('varO', 0):.6f}")

            if prog and fold not in completed_folds:
                any_active = True

            # Variance alerts
            if curve:
                alerts = check_variance_alert(curve)
                for a in alerts:
                    print(f"[{ts}] {a}")

        # Periodic summary every 60s
        if now - last_summary_time >= 60:
            last_summary_time = now
            print(f"\n[{ts}] --- Status ({len(procs)} train procs alive) ---")
            gpus = get_gpu_status()
            for g in gpus:
                if "err" not in g:
                    print(f"  GPU {g['idx']}: {g['util']:3d}%  mem={g['mem']:5d} MiB")

            for fold in range(5):
                fs = fold_summary[fold]
                if fs["best"]:
                    bar = get_progress_bar(fold)
                    if fs["progress"] and fold not in completed_folds:
                        # Last variance
                        curve = get_epoch_curve(fold)
                        if curve and len(curve) > 0:
                            last = curve[-1]
                            var_info = f"varW={last.get('varW', 0):.5f} varO={last.get('varO', 0):.5f}"
                        else:
                            var_info = ""
                        print(f"  Fold {fold}: best={fs['best']['cindex']:.4f}@{fs['best']['epoch']:2d} | {bar} | {var_info}")
                    else:
                        print(f"  Fold {fold}: ✓ DONE best={fs['best']['cindex']:.4f}@{fs['best']['epoch']:2d}")
            print()

        # Stop if all 5 folds done and no process
        if len(completed_folds) == 5 and not procs:
            print(f"\n[{ts}] 🏁 ALL 5 FOLDS COMPLETE. Monitor stopping.")
            break

        if not procs and not any_active:
            print(f"\n[{ts}] ⚠️ No training process running. Monitor exiting.")
            break

        time.sleep(30)

if __name__ == "__main__":
    main()
