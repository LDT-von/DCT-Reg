#!/usr/bin/env python3
"""实时监控 v3.11 5 折训练:从 epoch_curve_*.csv 读取每折 val_c 最佳/最新."""
import argparse, csv, os, sys, time

RESULTS_ROOT = "results/dct_v311_blca_uni_v2/blca/blca/SurvOTRank_dct_v311_slot_interpretable"
PARAMS_GLOB = "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_seed3_rW_8_rG_8_sp_dct_v311_blca_uni_v2"


def find_params_dir():
    p = os.path.join(RESULTS_ROOT, PARAMS_GLOB)
    return p if os.path.isdir(p) else None


def load_fold_curve(path):
    """返回 list of (epoch, val_c, val_ipcw, val_ibs, val_iauc)."""
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if not row:
                continue
            ep = int(row[0])
            val_c = float(row[1])
            val_ipcw = float(row[2])
            val_ibs = float(row[3])
            val_iauc = float(row[4])
            out.append((ep, val_c, val_ipcw, val_ibs, val_iauc))
    return out


def render(params_dir, current_fold, current_epoch, total_epochs, train_running):
    print("\033[2J\033[H", end="")
    print(f"[{time.strftime('%H:%M:%S')}] v3.11 BLCA UNI_v2 5 折监控", flush=True)
    if train_running:
        print(f"  ▶ 当前: Fold {current_fold}   Epoch {current_epoch}/{total_epochs}", flush=True)
    else:
        print(f"  ■ 训练已停止 (Fold {current_fold})", flush=True)
    print()
    print(f"  {'Fold':>4} | {'Best Ep':>7} | {'Best val_c':>10} | {'Latest val_c':>11} | {'val_ipcw':>8} | {'val_ibs':>7} | {'val_iauc':>7}", flush=True)
    print("  " + "-" * 75, flush=True)
    fold_files = sorted(
        [f for f in os.listdir(params_dir) if f.startswith("epoch_curve_fold") and f.endswith(".csv")],
        key=lambda x: int(x.replace("epoch_curve_fold", "").replace(".csv", "")),
    )
    best_overall = []
    for ff in fold_files:
        fold = int(ff.replace("epoch_curve_fold", "").replace(".csv", ""))
        rows = load_fold_curve(os.path.join(params_dir, ff))
        if not rows:
            print(f"  {fold:>4} | {'?':>7} | {'?':>10} | {'?':>11} | {'?':>8} | {'?':>7} | {'?':>7}", flush=True)
            continue
        latest = rows[-1]
        best = max(rows, key=lambda r: r[1])
        tag = " *" if (train_running and fold == current_fold) else ""
        print(
            f"  {fold:>4} | {best[0]:>7} | {best[1]:>10.4f} | {latest[1]:>11.4f} | {latest[2]:>8.4f} | {latest[3]:>7.4f} | {latest[4]:>7.4f}{tag}",
            flush=True,
        )
        best_overall.append(best[1])

    if best_overall:
        print()
        print(f"  ★ 已完成折 mean val_c = {sum(best_overall)/len(best_overall):.4f}  (n={len(best_overall)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=8.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--log", default="logs/dct_v311_blca_uni_v2_5fold.log")
    args = ap.parse_args()

    params_dir = find_params_dir()
    if params_dir is None:
        print("找不到结果目录:", RESULTS_ROOT + "/" + PARAMS_GLOB, file=sys.stderr)
        return 1

    last_mtime = 0
    while True:
        current_fold, current_epoch, total_epochs = -1, 0, 30
        train_running = False
        try:
            if os.path.exists(args.log):
                with open(args.log, "rb") as f:
                    data = f.read().decode("utf-8", errors="ignore")
                last_lines = data.splitlines()[-200:]
                for line in last_lines:
                    line = line.replace("\r", "")
                    if "Epoch" in line and "/" in line and "[" in line:
                        import re
                        m = re.search(r"\[Fold\s+(\d+)\].*Epoch\s+(\d+)/(\d+)", line)
                        if m:
                            current_fold = int(m.group(1))
                            current_epoch = int(m.group(2))
                            total_epochs = int(m.group(3))
                            train_running = True
                if not any("Epoch" in ln and "/" in ln for ln in last_lines):
                    train_running = False
        except Exception:
            train_running = False

        params_mtime = max(
            (os.path.getmtime(os.path.join(params_dir, f)) for f in os.listdir(params_dir) if f.startswith("epoch_curve_fold")),
            default=0,
        )
        log_mtime = os.path.getmtime(args.log) if os.path.exists(args.log) else 0
        if max(params_mtime, log_mtime) != last_mtime:
            last_mtime = max(params_mtime, log_mtime)
            render(params_dir, current_fold, current_epoch, total_epochs, train_running)
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main() or 0)
