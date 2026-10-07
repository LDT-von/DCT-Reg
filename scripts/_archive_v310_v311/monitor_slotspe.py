#!/usr/bin/env python3
"""
SlotSPE BLCA training monitor — 每折每 epoch 实时跟踪

由于 survival.py 的 stdout/log_file 是 buffered 的，本脚本采用：
  1. 监控 best_model mtime (val_cindex 提高时才更新)
  2. 监控 split_<fold>_results.pkl mtime (同上)
  3. 监控进程 CPU time 增量 (每个 epoch 约 +N 秒)
  4. 监控 GPU util
  5. 监控 internal log file 大小/mtime (fold 结束时会 flush)

Usage:
    python scripts/monitor_slotspe.py
    python scripts/monitor_slotspe.py --refresh 5 --json-out /tmp/slotspe_monitor.json
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


def get_process_info(pid):
    """返回 (etime_sec, cpu_time_sec)"""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        # 字段索引 (1-based in proc(5); 0-based in split):
        #   utime=14, stime=15, starttime=22, etime is NOT in stat; use uptime - starttime
        utime = int(parts[13])
        stime = int(parts[14])
        starttime_ticks = int(parts[21])
        clk_tck = os.sysconf("SC_CLK_TCK")
        cpu_sec = (utime + stime) / clk_tck
        # etime = uptime - starttime_ticks / clk_tck
        with open("/proc/uptime") as f:
            uptime_sec = float(f.read().split()[0])
        etime = max(0.0, uptime_sec - starttime_ticks / clk_tck)
        return etime, cpu_sec
    except Exception:
        return None, None


def get_gpu_info():
    """返回 (util%, mem_used_mib)"""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            timeout=5
        ).decode()
        lines = []
        for line in out.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 3:
                lines.append({"index": int(parts[0]),
                              "util": float(parts[1]),
                              "mem_mib": float(parts[2])})
        return lines
    except Exception:
        return []


def parse_log(log_path):
    """提取 epoch X 的 train_loss, train_c, val_c, val_c2, val_ibs, val_iauc."""
    rows = {}
    if not os.path.exists(log_path):
        return rows
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return rows
    train_re = re.compile(
        r"Epoch:\s*(\d+),\s*train_loss:\s*([\d.]+),\s*train_c_index:\s*([\d.]+)"
    )
    val_re = re.compile(
        r"Epoch:\s*(\d+)\s+Val c-index:\s*([\d.]+)\s*\|\s*Final Val c-index2:\s*([\d.]+)\s*\|\s*Final Val IBS:\s*([\d.]+)\s*\|\s*Final Val iauc:\s*([\d.]+)"
    )
    for m in train_re.finditer(text):
        ep = int(m.group(1))
        rows.setdefault(ep, {})["train_loss"] = float(m.group(2))
        rows[ep]["train_c"] = float(m.group(3))
    for m in val_re.finditer(text):
        ep = int(m.group(1))
        rows.setdefault(ep, {})
        rows[ep]["val_c"] = float(m.group(2))
        rows[ep]["val_c2"] = float(m.group(3))
        rows[ep]["val_ibs"] = float(m.group(4))
        rows[ep]["val_iauc"] = float(m.group(5))
    return rows


def discover_fold_dirs(results_dir):
    """发现每个 fold 的文件路径 (pkl, log, best_pth)."""
    pattern = os.path.join(results_dir, "blca", "SlotSPE", "*")
    exp_dirs = sorted(glob.glob(pattern))
    by_fold = {}
    for d in exp_dirs:
        pkl_files = sorted(glob.glob(os.path.join(d, "split_*_results.pkl")))
        if not pkl_files:
            continue
        settings_path = os.path.join(d, "experiment_settings.txt")
        k_start = k_end = None
        if os.path.exists(settings_path):
            try:
                with open(settings_path) as f:
                    text = f.read()
                m = re.search(r"'k_start':\s*(\d+).*?'k_end':\s*(\d+)", text, re.DOTALL)
                if m:
                    k_start = int(m.group(1))
                    k_end = int(m.group(2))
            except Exception:
                pass
        if k_start is None:
            fold = int(pkl_files[0].split("split_")[1].split("_")[0])
            k_start, k_end = fold, fold + 1
        for fold in range(k_start, k_end):
            pkl = os.path.join(d, f"split_{fold}_results.pkl")
            log = os.path.join(d, f"log_start_{k_start}_end_{k_end}.txt")
            best_pth = os.path.join(d, f"model_best_s{fold}.pth")
            by_fold[fold] = {"pkl": pkl, "log": log, "best_pth": best_pth, "params_dir": d}
    return by_fold


def fmt_mtime(path):
    if os.path.exists(path):
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%H:%M:%S")
    return "    -"


def print_state(state, refresh, json_path=None):
    by_fold = state["by_fold"]
    pid = state["pid"]
    etime = state.get("etime_sec", 0)
    cpu = state.get("cpu_sec", 0)
    gpus = state["gpus"]

    n_folds = max(by_fold.keys()) + 1 if by_fold else 0
    # 收集每个 fold 所有 epoch 数据
    rows_by_fold = state["rows_by_fold"]

    # 找最大 epoch 数
    max_ep = 0
    for f_rows in rows_by_fold.values():
        if f_rows:
            max_ep = max(max_ep, max(f_rows.keys()))

    # 表头
    headers = ["Ep"]
    for f in range(n_folds):
        headers.append(f"f{f}.val_c")
    for f in range(n_folds):
        headers.append(f"f{f}.val_ibs")
    for f in range(n_folds):
        headers.append(f"f{f}.iauc")
    col_w = [6] + [12] * (len(headers) - 1)
    sep = "+".join("-" * w for w in col_w)
    sep = f"+{sep}+"

    lines = []
    lines.append(f"SlotSPE BLCA monitor @ {datetime.now().strftime('%H:%M:%S')}  "
                 f"(refresh={refresh}s, pid={pid} etime={int(etime//60)}m{int(etime%60)}s cpu={int(cpu//60)}m{int(cpu%60)}s)")
    gpu_str = " | ".join(f"gpu{g['index']}: {g['util']:.0f}% {g['mem_mib']:.0f}MiB" for g in gpus)
    lines.append(f"GPU: {gpu_str}")
    lines.append(sep)
    lines.append("|" + "|".join(f" {h:^{w-2}} " for h, w in zip(headers, col_w)) + "|")
    lines.append(sep)
    for ep in range(max_ep + 1):
        cells = [f"{ep:>{col_w[0]-2}}"]
        for f in range(n_folds):
            row = rows_by_fold.get(f, {}).get(ep, {})
            v = row.get("val_c")
            cells.append(f"{v:>{col_w[1+f]-2}.4f}" if v is not None else f"{'.':>{col_w[1+f]-2}}")
        for f in range(n_folds):
            row = rows_by_fold.get(f, {}).get(ep, {})
            v = row.get("val_ibs")
            cells.append(f"{v:>{col_w[1+n_folds+f]-2}.4f}" if v is not None else f"{'.':>{col_w[1+n_folds+f]-2}}")
        for f in range(n_folds):
            row = rows_by_fold.get(f, {}).get(ep, {})
            v = row.get("val_iauc")
            cells.append(f"{v:>{col_w[1+2*n_folds+f]-2}.4f}" if v is not None else f"{'.':>{col_w[1+2*n_folds+f]-2}}")
        lines.append("|" + "|".join(f" {c} " for c in cells) + "|")
    lines.append(sep)

    # 每 fold 状态 (pkl mtime / best mtime / best val_c / best epoch)
    lines.append("")
    lines.append("FOLD STATUS:")
    lines.append("-" * 78)
    for f in sorted(by_fold.keys()):
        info = by_fold[f]
        pkl_mt = fmt_mtime(info["pkl"]) if info["pkl"] else "    -"
        pth_mt = fmt_mtime(info["best_pth"]) if info["best_pth"] else "    -"
        # best val_c from parsed log
        rows = rows_by_fold.get(f, {})
        if rows:
            best_ep = max(rows.keys(), key=lambda e: rows[e].get("val_c", -1))
            best_row = rows[best_ep]
            best_v = best_row.get("val_c")
            best_v_ibs = best_row.get("val_ibs")
            best_v_iauc = best_row.get("val_iauc")
            best_str = f"best val_c={best_v:.4f}@{best_ep} (ibs={best_v_ibs:.4f} iauc={best_v_iauc:.4f})" if best_v else "no val yet"
        else:
            best_str = "log not flushed yet"
        lines.append(f"  fold{f}: pkl_mtime={pkl_mt}  best_model_mtime={pth_mt}  {best_str}")
    lines.append("-" * 78)

    # 事件
    if state.get("events"):
        lines.append("EVENTS:")
        for ev in state["events"][-10:]:
            lines.append(f"  {ev}")

    out = "\n".join(lines)
    print(out)

    # 写 JSON
    if json_path:
        try:
            json_state = {
                "ts": datetime.now().isoformat(),
                "pid": pid,
                "etime_sec": etime,
                "cpu_sec": cpu,
                "gpus": gpus,
                "rows_by_fold": {f: {ep: rows_by_fold[f][ep] for ep in rows_by_fold[f]}
                                 for f in rows_by_fold},
                "fold_files": {f: {k: by_fold[f][k] for k in by_fold[f]} for f in by_fold},
            }
            with open(json_path, "w") as f:
                json.dump(json_state, f, indent=2, default=str)
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/data1/DCT-Reg/results/slotspe_blca_paper")
    ap.add_argument("--pid", type=int, default=3293668,
                    help="SlotSPE training process PID")
    ap.add_argument("--refresh", type=float, default=10.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    last_pkl_mtime = {}
    last_pth_mtime = {}
    last_log_mtime = {}
    last_log_size = {}
    events = []
    rows_by_fold = {}

    started = datetime.now()
    while True:
        if not os.path.isdir(args.results_dir):
            print(f"Results dir {args.results_dir} not found yet. Waiting...")
            time.sleep(args.refresh)
            continue

        # 探索 fold 目录
        by_fold = discover_fold_dirs(args.results_dir)
        # 解析 logs
        for f, info in by_fold.items():
            log = info["log"]
            if os.path.exists(log):
                mt = os.path.getmtime(log)
                sz = os.path.getsize(log)
                if f not in last_log_mtime or mt != last_log_mtime[f] or sz != last_log_size.get(f):
                    events.append(f"[{datetime.now().strftime('%H:%M:%S')}] fold{f} log updated (size={sz})")
                    last_log_mtime[f] = mt
                    last_log_size[f] = sz
                rows_by_fold[f] = parse_log(log)

        # pkl mtime 跟踪
        for f, info in by_fold.items():
            if info["pkl"] and os.path.exists(info["pkl"]):
                mt = os.path.getmtime(info["pkl"])
                if f in last_pkl_mtime and mt != last_pkl_mtime[f]:
                    events.append(f"[{datetime.now().strftime('%H:%M:%S')}] fold{f} BEST UPDATED (pkl mtime changed)")
                last_pkl_mtime[f] = mt
            if info["best_pth"] and os.path.exists(info["best_pth"]):
                mt = os.path.getmtime(info["best_pth"])
                if f in last_pth_mtime and mt != last_pth_mtime[f]:
                    events.append(f"[{datetime.now().strftime('%H:%M:%S')}] fold{f} BEST MODEL UPDATED")
                last_pth_mtime[f] = mt

        # CPU/GPU
        etime_sec, cpu_sec = get_process_info(args.pid)
        gpus = get_gpu_info()

        state = {
            "pid": args.pid,
            "etime_sec": etime_sec or 0,
            "cpu_sec": cpu_sec or 0,
            "gpus": gpus,
            "by_fold": by_fold,
            "rows_by_fold": rows_by_fold,
            "events": events,
        }

        # 清屏输出
        try:
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()
        except Exception:
            pass
        elapsed = datetime.now() - started
        print(f"⏱  SlotSPE monitor (running for {elapsed})  refresh={args.refresh}s  Ctrl-C to quit\n")
        print_state(state, args.refresh, args.json_out)

        time.sleep(args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBye")
