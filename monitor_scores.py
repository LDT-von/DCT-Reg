#!/usr/bin/env python3
"""每个 epoch 分数 + 每折 best 的实时监控。

结构: <results>/<某级>/<study>/<method>/<exp>/epoch_curve_fold{N}.csv

用法:
  python monitor_scores.py                 # 自动找 root 下"最新写入"的实验
  python monitor_scores.py --root <results目录>
  python monitor_scores.py --exp <实验绝对路径>
  python monitor_scores.py --all           # 列出 root 下所有实验
  python monitor_scores.py -w 60           # 每 60s 刷新, 实时监控 (Ctrl-C 退出)
  python monitor_scores.py -m val_iauc     # 换关注指标
"""
import argparse
import csv
import glob
import os
import sys
import time


def find_experiments(root):
    """返回 {exp_dir: {fold: (csv_path, mtime)}}"""
    exps = {}
    pattern = os.path.join(root, "**", "epoch_curve_fold*.csv")
    for path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        try:
            fold = int(name.replace("epoch_curve_fold", "").replace(".csv", ""))
        except ValueError:
            continue
        exp_dir = os.path.dirname(path)
        exps.setdefault(exp_dir, {})[fold] = (path, os.path.getmtime(path))
    return exps


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "  -  "


def print_fold_table(rows, metric, metric2, best_ep):
    print(f"      ep | {metric:>10} | {metric2:>9} | best")
    print("      " + "-" * 42)
    for r in rows:
        ep = int(float(r["epoch"]))
        star = " *" if ep == best_ep else "  "
        print(
            f"      {ep:>3} | {fmt(r.get(metric)):>10} | {fmt(r.get(metric2)):>9} | {star}"
        )


def analyze_fold(path, metric, metric2):
    rows = load_rows(path)
    if not rows:
        return None
    best = None
    best_ep = None
    for r in rows:
        v = float(r.get(metric))
        ep = int(float(r["epoch"]))
        if best is None or v > best:
            best = v
            best_ep = ep
    return {"rows": rows, "best": best, "best_ep": best_ep}


def report_one(exp_dir, metric, metric2, expected_epochs, show_detail):
    folds = {}
    for p in glob.glob(os.path.join(exp_dir, "epoch_curve_fold*.csv")):
        try:
            f = int(os.path.basename(p).replace("epoch_curve_fold", "").replace(".csv", ""))
        except ValueError:
            continue
        folds[f] = p
    if not folds:
        print(f"[等待] 尚无 epoch_curve_fold*.csv: {exp_dir}")
        return

    newest_path, newest_mtime = max(
        ((p, os.path.getmtime(p)) for p in folds.values()), key=lambda x: x[1])
    age = time.time() - newest_mtime

    print(f"\n实验目录: {exp_dir}")
    print(f"  监控指标: {metric}   (最新写入 {int(age)}s 前)")

    per_fold_best = []
    for f in sorted(folds):
        path = folds[f]
        info = analyze_fold(path, metric, metric2)
        if info is None:
            continue
        rows = info["rows"]
        done = len(rows) >= expected_epochs
        status = "完成" if done else "进行中"
        per_fold_best.append((f, info["best"], info["best_ep"], status))

        if show_detail:
            print(f"\n  [Fold {f}]  {status}  ({len(rows)}/{expected_epochs} epochs)")
            print_fold_table(rows, metric, metric2, info["best_ep"])

    print(f"\n  {'折':<6}{'best_ep':<9}{'best_'+metric:<16}{'最新ep值':<12}{'状态':<6}")
    print("  " + "-" * 54)
    for f, best, best_ep, status in per_fold_best:
        rows = load_rows(folds[f])
        last = float(rows[-1].get(metric))
        print(f"  Fold {f}   ep {best_ep:<6}  {fmt(best):<16}  {fmt(last):<12}  {status}")
    if per_fold_best:
        avg = sum(b for _, b, _, _ in per_fold_best) / len(per_fold_best)
        best_all = max(per_fold_best, key=lambda x: x[1])
        print(f"\n  已完成折平均 best_{metric}: {avg:.4f}  |  当前最优: "
              f"Fold {best_all[0]} ep {best_all[2]} = {best_all[1]:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data1/DCT-Reg/results")
    ap.add_argument("--exp", default=None)
    ap.add_argument("--only", default=None,
                    help="只看指定实验(逗号分隔,按路径子串匹配), 可多个并排显示, 如: dct_v311_blca_uni2h,dct_v311_blca_uni")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("-w", "--watch", type=float, default=0.0)
    ap.add_argument("-m", "--metric", default="val_cindex")
    ap.add_argument("-m2", "--metric2", default="val_iauc")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--no-detail", action="store_true")
    args = ap.parse_args()

    if args.exp:
        targets = [args.exp]
    elif args.only:
        tokens = [t.strip() for t in args.only.split(",") if t.strip()]
        exps = find_experiments(args.root)
        targets = []
        for t in tokens:
            hits = [d for d in exps
                    if os.path.relpath(d, args.root).split(os.sep)[0] == t]
            if hits:
                targets.append(max(hits, key=lambda d: max(m for _, m in exps[d].values())))
            else:
                targets.append(os.path.join(args.root, t))  # 尚未开始写 CSV, report 会显示等待
    else:
        exps = find_experiments(args.root)
        if not exps:
            print(f"在 {args.root} 下没找到任何 epoch_curve_fold*.csv")
            sys.exit(1)
        ordered = sorted(exps.items(), key=lambda kv: -max(m for _, m in kv[1].values()))
        if args.all:
            print(f"共 {len(ordered)} 个实验:")
            for i, (d, _) in enumerate(ordered):
                print(f"  [{i}] {d}")
            targets = [d for d, _ in ordered]
        else:
            targets = [ordered[0][0]]

    try:
        while True:
            if args.watch > 0:
                os.system("clear")
            for t in targets:
                report_one(t, args.metric, args.metric2, args.epochs,
                           not args.no_detail)
            if args.watch <= 0:
                break
            time.sleep(args.watch)
    except KeyboardInterrupt:
        print("\n退出监控")


if __name__ == "__main__":
    main()
