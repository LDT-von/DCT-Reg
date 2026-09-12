"""Comprehensive scan: every v3.10/v3.11 candidate dir, compute 5-fold best val_cindex mean."""
import os
import csv
import json

ROOT = "/data1/DCT-Reg/results"

def load_curve(path):
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                out.append((int(row["epoch"]), float(row["val_cindex"])))
            except (KeyError, ValueError):
                continue
    return out

def best_per_fold(curve):
    if not curve:
        return None
    return max(v for _, v in curve)

def collect(exp_dir):
    curves = sorted(f for f in os.listdir(exp_dir) if f.startswith("epoch_curve_fold") and f.endswith(".csv"))
    if not curves:
        return None
    fold_bests = []
    for cf in curves:
        c = load_curve(os.path.join(exp_dir, cf))
        b = best_per_fold(c)
        if b is not None:
            fold_bests.append((cf, b))
    if not fold_bests:
        return None
    mean = sum(v for _, v in fold_bests) / len(fold_bests)
    return mean, fold_bests

def find_exp_dirs(root, name_substr):
    """Return all dirs that contain epoch_curve_fold*.csv and name_substr in path."""
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        if any(f.startswith("epoch_curve_fold") for f in filenames):
            if name_substr in dirpath:
                found.append(dirpath)
    return sorted(found)

def main():
    print("=" * 100)
    print("ALL v3.10 candidates (by 5-fold best val_cindex mean)")
    print("=" * 100)
    for d in find_exp_dirs(ROOT, "v310"):
        rel = d.replace(ROOT + "/", "")
        res = collect(d)
        if res is None:
            print(f"{rel}: NO FOLD CURVES")
            continue
        mean, folds = res
        fold_str = ", ".join(f"{name.replace('epoch_curve_','').replace('.csv','')}={v:.4f}" for name, v in folds)
        print(f"  mean={mean:.4f}  ({len(folds)} folds)  [{rel}]")
        print(f"      {fold_str}")

    print()
    print("=" * 100)
    print("ALL v3.11 candidates (by 5-fold best val_cindex mean)")
    print("=" * 100)
    for d in find_exp_dirs(ROOT, "v311"):
        rel = d.replace(ROOT + "/", "")
        res = collect(d)
        if res is None:
            print(f"{rel}: NO FOLD CURVES")
            continue
        mean, folds = res
        fold_str = ", ".join(f"{name.replace('epoch_curve_','').replace('.csv','')}={v:.4f}" for name, v in folds)
        print(f"  mean={mean:.4f}  ({len(folds)} folds)  [{rel}]")
        print(f"      {fold_str}")

if __name__ == "__main__":
    main()