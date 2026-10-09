#!/usr/bin/env python3
"""Batch re-export v3.13 exp6 evidence with km=True to recover
km_train_median in export.json. Required by survot_rank.evidence.km_oof
to pool validation risk groups across the five folds.

Different from rerun_v313_export_for_attention.py:
- That script uses km=False (only restores attention_omic).
- This one uses km=True so km_train_median is recorded and
  plot_fig5_km_curves.py can produce pooled KM figures.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = "/home/ubuntu/.conda/envs/trisurv/bin/python"

MANIFESTS = [
    REPO_ROOT / "results/v313_evidence_v2/full_manifest.json",
    REPO_ROOT / "results/v313_evidence_v2/controls.json",
]

EXPORT_ROOT = REPO_ROOT / "results/v313_interpretability_v1/exports"


def collect_runs(arm=None, cancer=None, fold=None):
    out = []
    for mf in MANIFESTS:
        if not mf.exists():
            print(f"[warn] manifest not found: {mf}")
            continue
        manifest = json.load(open(mf))
        for run in manifest["runs"]:
            if arm and run.get("arm") != arm:
                continue
            if cancer and run.get("cancer") != cancer:
                continue
            if fold is not None and run.get("fold") != fold:
                continue
            out.append((mf, run))
    return out


def backup_old(run_id: str):
    old = EXPORT_ROOT / run_id
    if old.exists():
        backup = old.with_name(old.name + ".bak_km_export")
        if not backup.exists():
            old.rename(backup)
            return backup
        else:
            import shutil
            shutil.rmtree(str(old))
            return backup


def export_one(run, device):
    run_id = run["id"]
    backup = backup_old(run_id)
    if backup:
        print(f"  [backup] {run_id} -> {backup.name}", flush=True)

    out_dir = EXPORT_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT))
    from survot_rank.evidence.v313 import export_run

    t0 = time.time()
    try:
        export_run(run, str(EXPORT_ROOT),
                   device=device,
                   alphas=(0, 0.25, 0.5, 0.75, 1),
                   case_ids=(),
                   km=True, profile=False)
        elapsed = time.time() - t0
        # Verify km_train_median in the new export.json
        meta = json.loads((EXPORT_ROOT / run_id / "export.json").read_text())
        ktm = meta.get("km_train_median")
        return {"ok": True, "elapsed": elapsed, "km_train_median": ktm}
    except Exception as e:
        elapsed = time.time() - t0
        return {"ok": False, "elapsed": elapsed, "error": str(e)}


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument("--arm", choices=["exp6", "direct", "independent"])
    ap.add_argument("--cancer", choices=["blca", "kirc"])
    ap.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    runs = collect_runs(arm=args.arm, cancer=args.cancer, fold=args.fold)
    print(f"Found {len(runs)} runs", flush=True)

    n_ok = n_err = 0
    results = []
    for mf, r in runs:
        ck = r.get("checkpoint", "?")
        if not Path(ck).exists():
            print(f"[{r['arm']}/{r['cancer']}/fold{r['fold']}] {r['id']}: checkpoint missing", flush=True)
            results.append({"id": r["id"], "ok": False, "error": "checkpoint missing"})
            n_err += 1
            continue
        print(f"\n[{r['arm']}/{r['cancer']}/fold{r['fold']}] {r['id']}", flush=True)
        result = export_one(r, args.device)
        result["id"] = r["id"]
        results.append(result)
        if result["ok"]:
            n_ok += 1
            print(f"  [OK] {result['elapsed']:.0f}s  km_train_median={result.get('km_train_median')}", flush=True)
        else:
            n_err += 1
            print(f"  [ERROR] {result.get('error', '?')}", flush=True)

    print(f"\n{'='*60}\n  Results: {n_ok} OK, {n_err} failed\n{'='*60}")
    Path("logs/rerun_v313_export_km.json").write_text(json.dumps(results, indent=2, default=str))
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())