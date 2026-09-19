#!/usr/bin/env python3
"""Summarize v3.11 uni Fixed pan-cancer training: extract best cindex
per fold per cancer from the per-fold logs and compute mean ± std.
"""
import re
from pathlib import Path
from collections import defaultdict

LOG_ROOT = Path("/data1/DCT-Reg/logs/v311_blca_uni_fixed_pancancer")
BEST_PATTERN = re.compile(r"\[Fold (\d+)\] best cindex=([\d.]+) @epoch (\d+)")

# cancer | fold -> best cindex
results: dict[str, dict[int, float]] = defaultdict(dict)

for log_file in sorted(LOG_ROOT.glob("*_fold*.log")):
    name = log_file.stem  # e.g. brca_fold0
    if "_fold" not in name:
        continue
    cancer, fold_s = name.rsplit("_fold", 1)
    try:
        fold = int(fold_s)
    except ValueError:
        continue
    text = log_file.read_text()
    matches = BEST_PATTERN.findall(text)
    if matches:
        # 选最后一个 best（每个 fold 训练结束只有一个 best cindex）
        best = float(matches[-1][1])
        results[cancer][fold] = best

if not results:
    print("No completed folds yet.")
    raise SystemExit(0)

# 输出汇总
print("=" * 75)
print(" v3.11 UNI FIXED — Pan-cancer 5-fold C-index")
print("=" * 75)
print(f"{'Cancer':<12} {'Mean':>8} {'Std':>8} {'Folds'}")
print("-" * 75)

summary = []
for cancer in sorted(results):
    folds = results[cancer]
    if not folds:
        continue
    vals = sorted(folds.items())
    values = [v for _, v in vals]
    mean = sum(values) / len(values)
    var = sum((x - mean) ** 2 for x in values) / len(values)
    std = var ** 0.5
    summary.append((cancer, mean, std, vals))
    folds_str = ", ".join([f"f{f}={v:.4f}" for f, v in vals])
    print(f"{cancer:<12} {mean:.4f}   {std:.4f}   {folds_str}")

print("-" * 75)
# 排名
summary.sort(key=lambda x: -x[1])
print("\nRanked by mean C-index:")
for i, (cancer, m, s, vals) in enumerate(summary, 1):
    print(f"  {i}. {cancer:<10} {m:.4f} ± {s:.4f}  ({len(vals)}/5 folds done)")
