#!/usr/bin/env python3
"""实时监控 SlotSPE 多折训练,只输出每折当前最佳 val_c 与最新指标。"""
import json, os, sys, time, argparse

DEFAULT_STATE = "logs/slotspe_blca_paper/monitor_state.json"


def load_state(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def fold_best(rows):
    """返回 (best_epoch, best_val_c, latest_epoch, latest_row)."""
    if not rows:
        return None, None, None, None
    eps = sorted(rows.items(), key=lambda x: int(x[0]))
    latest_ep, latest_row = eps[-1]
    done = [(int(ep), r) for ep, r in eps if r.get("val_c") is not None]
    if not done:
        return None, None, int(latest_ep), latest_row
    best_ep, best_row = max(done, key=lambda x: x[1]["val_c"])
    return int(best_ep), best_row["val_c"], int(latest_ep), latest_row


def render(state, show_running=True):
    if state is None:
        print(f"[{time.strftime('%H:%M:%S')}] 等待 monitor_state.json ...", flush=True)
        return

    etime = int(state.get("etime_sec", 0))
    h, rem = divmod(etime, 3600)
    m, s = divmod(rem, 60)
    print(f"[{time.strftime('%H:%M:%S')}] 已运行 {h:02d}:{m:02d}:{s:02d} | cpu={int(state.get('cpu_sec', 0))}s", flush=True)

    rows_by_fold = state.get("rows_by_fold", {})
    if not rows_by_fold:
        print("  暂无 fold 数据", flush=True)
        return

    print(f"  {'Fold':>4} | {'Epochs':>6} | {'Best Ep':>7} | {'Best val_c':>10} | {'Latest val_c':>11} | {'val_ibs':>7} | {'val_iauc':>7}", flush=True)
    print("  " + "-" * 78, flush=True)
    for f in sorted(rows_by_fold.keys(), key=lambda x: int(x)):
        rows = rows_by_fold[f]
        best_ep, best_c, latest_ep, latest = fold_best(rows)
        if best_ep is None:
            if show_running:
                print(f"  {f:>4} | {len(rows):>6} | {'?':>7} | {'?':>10} | {('ep'+str(latest_ep)) if latest_ep else '?':>11} | {'?':>7} | {'?':>7}", flush=True)
            continue
        status = "" if latest_ep >= int(sorted(rows.keys(), key=lambda x: int(x))[-1]) or "val_c" in latest else " (训练中)"
        # 区分"完成的 fold"和"还在跑的 fold":最新 epoch 没有 val_c 就是还在训练
        latest_has_val = latest.get("val_c") is not None
        tag = "" if latest_has_val else " *"
        print(f"  {f:>4} | {len(rows):>6} | {best_ep:>7} | {best_c:>10.4f} | {latest['val_c']:>11.4f} | {latest['val_ibs']:>7.4f} | {latest['val_iauc']:>7.4f}{tag}", flush=True)

    # 如果有官方 summary CSV,顺便显示
    fold_files = state.get("fold_files", {})
    summary_csvs = []
    for fold, files in fold_files.items():
        params_dir = files.get("params_dir", "")
        for f in sorted(os.listdir(params_dir)):
            if f.startswith("summary_partial_") and f.endswith(".csv"):
                summary_csvs.append((fold, os.path.join(params_dir, f)))
    if summary_csvs:
        print("\n  官方 summary CSV (已写盘的):", flush=True)
        for fold, path in summary_csvs:
            print(f"    fold{fold}: {os.path.basename(path)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=DEFAULT_STATE)
    ap.add_argument("--interval", type=float, default=10.0, help="刷新间隔(秒)")
    ap.add_argument("--once", action="store_true", help="只跑一次就退出")
    args = ap.parse_args()

    last_mtime = 0
    while True:
        if os.path.exists(args.path):
            mtime = os.path.getmtime(args.path)
            if mtime != last_mtime:
                last_mtime = mtime
                state = load_state(args.path)
                # 清屏(ANSI),简洁观察
                sys.stdout.write("\033[2J\033[H")
                render(state)
        else:
            sys.stdout.write("\033[2J\033[H")
            render(None)

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
