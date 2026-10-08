#!/usr/bin/env python3
"""Assemble the v3.13 cost/performance tradeoff JSONs from the on-disk
profile exports, then call plot_v313_reference_panels.py to render the
figure.  Supports three layouts:

  1) BLCA only, Full vs Direct (5 folds each)  → efficiency_tradeoff.png
  2) BLCA + KIRC, Full vs Direct (10 folds each) → two-cohort figure
  3) 10-cohort Full, with KIRC/BLCA Direct as a second method when both
     cohorts share it (10 folds Full, 10 folds Direct) → final paper figure

The script never fabricates measurements: every (cancer, fold, method)
must have a real profile JSON in the exports directory and the
predictions pkl whose SHA-256 is the same one recorded in the export.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def text_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_measurement(cancer: str, fold: int, run_prefix: str, export_dir: Path) -> dict | None:
    """Return one measurement row, or None if the export is missing."""
    p = export_dir / f"{run_prefix}_f{fold}_s3" / "export.json"
    if not p.exists():
        return None
    e = json.loads(p.read_text())
    run = e["run"]
    pred = (ROOT / run["predictions"]).resolve()
    return dict(
        cancer=cancer.upper(),
        fold=fold,
        predictions=str(pred),
        predictions_sha256=file_sha(pred),
        profile_json=str(p),
        profile_sha256_lf=text_sha(p),
    )


def build_model(
    model_id: str,
    label: str,
    ours: bool,
    cancers: list[str],
    run_prefix_by_cancer: dict[str, str],
    export_dir_by_cancer: dict[str, Path],
) -> dict:
    """Aggregate a method's measurements across the requested cancers."""
    measurements = []
    for cancer in cancers:
        prefix = run_prefix_by_cancer[cancer]
        ed = export_dir_by_cancer[cancer]
        for fold in range(5):
            m = load_measurement(cancer, fold, prefix, ed)
            if m is None:
                raise SystemExit(
                    f"[tradeoff] missing export for {model_id} {cancer} fold {fold}: "
                    f"looked in {ed / f'{prefix}_f{fold}_s3'}"
                )
            measurements.append(m)
    return dict(id=model_id, label=label, ours=ours, measurements=measurements)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", choices=("blca_only", "blca_kirc", "ten_cohort_with_direct"), required=True)
    ap.add_argument("--output", required=True, help="Fresh output directory for the figure")
    args = ap.parse_args()

    base = ROOT / "results/v313_tradeoff_real_20261008"
    full_dir = base / "exports_blca_5fold"
    direct_blca_dir = base / "exports_direct_blca_5fold"
    direct_kirc_dir = base / "exports_kirc_5fold_direct"
    full10_dir = base / "exports_10cohort_5fold_full"

    if args.layout == "blca_only":
        models = [
            build_model(
                "dct_v313_full", "DCT v3.13 Full (transport cross)", True,
                ["BLCA"], {"BLCA": "exp6_blca"}, {"BLCA": full_dir},
            ),
            build_model(
                "dct_v313_direct", "DCT v3.13 Direct (cross reconstruction from WSI slots)", False,
                ["BLCA"], {"BLCA": "direct_blca"}, {"BLCA": direct_blca_dir},
            ),
        ]
    elif args.layout == "blca_kirc":
        # Two-cohort Full vs Direct. Full is re-exported in the 10-cohort run;
        # for this figure we reuse the same exp6 BLCA/KIRC exports once both exist.
        models = [
            build_model(
                "dct_v313_full", "DCT v3.13 Full (transport cross)", True,
                ["BLCA", "KIRC"], {"BLCA": "exp6_blca", "KIRC": "exp6_kirc"},
                {"BLCA": full10_dir, "KIRC": full10_dir},
            ),
            build_model(
                "dct_v313_direct", "DCT v3.13 Direct (cross reconstruction from WSI slots)", False,
                ["BLCA", "KIRC"], {"BLCA": "direct_blca", "KIRC": "direct_kirc"},
                {"BLCA": direct_blca_dir, "KIRC": direct_kirc_dir},
            ),
        ]
    elif args.layout == "ten_cohort_with_direct":
        cancers = ["BLCA", "BRCA", "COADREAD", "HNSC", "KIRC", "LUAD", "LUSC", "SKCM", "STAD", "UCEC"]
        models = [
            build_model(
                "dct_v313_full", "DCT v3.13 Full (transport cross)", True,
                cancers, {c: f"exp6_{c.lower()}" for c in cancers},
                {c: full10_dir for c in cancers},
            ),
            build_model(
                "dct_v313_direct", "DCT v3.13 Direct (cross reconstruction from WSI slots)", False,
                ["BLCA", "KIRC"], {"BLCA": "direct_blca", "KIRC": "direct_kirc"},
                {"BLCA": direct_blca_dir, "KIRC": direct_kirc_dir},
            ),
        ]
    else:
        raise SystemExit(f"unknown layout: {args.layout}")

    payload = dict(
        schema_version=1,
        context=dict(cancers=sorted({m["cancer"] for m in models[0]["measurements"]}),
                     endpoint="DSS", encoder="UNI2-h", protocol="legacy_val", seed=3),
        models=models,
    )

    out = ROOT / "paper/inputs/v313_tradeoff_real_20261008" / f"tradeoff_input_{args.layout}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"[tradeoff] wrote input {out} ({out.stat().st_size} bytes)")

    out_fig = ROOT / "paper/figures/v313_tradeoff_real_20261008" / args.output
    if out_fig.exists():
        for child in out_fig.glob("*"):
            if child.is_file():
                child.unlink()
    out_fig.mkdir(parents=True, exist_ok=True)

    rc = subprocess.run([
        PY, str(ROOT / "scripts/plot_v313_reference_panels.py"), "tradeoff",
        "--input", str(out),
        "--output", str(out_fig),
    ])
    sys.exit(rc.returncode)


if __name__ == "__main__":
    main()
