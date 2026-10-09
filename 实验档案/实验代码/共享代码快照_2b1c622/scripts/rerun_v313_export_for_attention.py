#!/usr/bin/env python3
"""
Batch re-export v3.13 evidence so `attention_omic` is included in
`patients.npz`. The original export script accidentally excluded both
`attention_wsi` and `attention_omic`; we keep WSI excluded (too large,
[N, Kw, 2048]) but include omic so the cohort pathway heatmap can use
real attention instead of self_pathway_error (reconstruction error).

The old export directories are renamed to <id>.bak_attention_omic_fix
so we can compare or roll back.

Usage:
    # Plan only (default)
    python scripts/rerun_v313_export_for_attention.py

    # Execute on all runs
    python scripts/rerun_v313_export_for_attention.py --run

    # Subset
    python scripts/rerun_v313_export_for_attention.py --run --cancer blca
    python scripts/rerun_v313_export_for_attention.py --run --arm exp6
    python scripts/rerun_v313_export_for_attention.py --run --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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
BACKUP_SUFFIX = ".bak_attention_omic_fix"


def collect_runs(cancer=None, arm=None, fold=None):
    out = []
    for mf in MANIFESTS:
        if not mf.exists():
            print(f"[warn] manifest not found: {mf}")
            continue
        manifest = json.load(open(mf))
        for run in manifest["runs"]:
            if cancer and run.get("cancer") != cancer:
                continue
            if arm and run.get("arm") != arm:
                continue
            if fold is not None and run.get("fold") != fold:
                continue
            out.append((mf, run))
    return out


def backup_old(run_id: str) -> Path | None:
    old = EXPORT_ROOT / run_id
    if not old.exists():
        return None
    backup = old.with_name(old.name + BACKUP_SUFFIX)
    if backup.exists():
        return backup
    shutil.move(str(old), str(backup))
    return backup


def verify_attention_omic(run_id: str) -> bool:
    """Check that patients.npz now contains attention_omic."""
    pat_path = EXPORT_ROOT / run_id / "patients.npz"
    if not pat_path.exists():
        return False
    try:
        import numpy as np
        data = np.load(pat_path, allow_pickle=True)
        return "attention_omic" in data.files
    except Exception:
        return False


def export_one(run: dict, device: str) -> dict:
    """Re-export a single run, capture results."""
    run_id = run["id"]
    backup = backup_old(run_id)
    if backup:
        print(f"  [backup] {run_id} -> {backup.name}")

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
                   km=False, profile=False)
        elapsed = time.time() - t0
        has_attn = verify_attention_omic(run_id)
        return {"ok": True, "elapsed": elapsed, "has_attention_omic": has_attn}
    except Exception as e:
        elapsed = time.time() - t0
        return {"ok": False, "elapsed": elapsed, "error": str(e)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="store_true", help="Execute the exports (default: plan only)")
    ap.add_argument("--cancer", choices=["blca", "kirc"])
    ap.add_argument("--arm", choices=["exp6", "direct", "independent"])
    ap.add_argument("--fold", type=int, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--log", default="logs/rerun_v313_export_attention.log")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)

    runs = collect_runs(cancer=args.cancer, arm=args.arm, fold=args.fold)
    print(f"Found {len(runs)} runs matching filters\n")
    for mf, r in runs:
        ckpt = r.get("checkpoint", "?")
        exists = "✓" if Path(ckpt).exists() else "✗"
        print(f"  [{exists}] {r['arm']:11s} {r['cancer']:4s} fold{r['fold']}  {r['id']}")

    if not args.run:
        print("\n[plan only] Re-run with --run to execute.")
        return 0

    # Execute
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    n_ok = n_err = 0
    for mf, r in runs:
        print(f"\n[{r['arm']}/{r['cancer']}/fold{r['fold']}] {r['id']}")
        if not Path(r.get("checkpoint", "")).exists():
            print(f"  [skip] checkpoint missing: {r['checkpoint']}")
            results.append({"id": r["id"], "ok": False, "error": "checkpoint missing"})
            n_err += 1
            continue

        result = export_one(r, device=args.device)
        result["id"] = r["id"]
        results.append(result)
        if result["ok"]:
            n_ok += 1
            attn_flag = "✓ has attention_omic" if result.get("has_attention_omic") else "✗ MISSING"
            print(f"  [OK] {result['elapsed']:.1f}s  {attn_flag}")
        else:
            n_err += 1
            print(f"  [ERROR] {result.get('error', '?')}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Results: {n_ok} OK, {n_err} failed (of {len(runs)})")
    print(f"{'='*60}")

    log_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Detailed log: {log_path}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
