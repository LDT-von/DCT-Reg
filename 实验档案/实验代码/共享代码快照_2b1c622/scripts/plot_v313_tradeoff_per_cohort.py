#!/usr/bin/env python3
"""Render a per-cohort scatter of v3.13 Full cost/performance: one
point per cancer (10 cohorts × 5-fold macro-average).  Supplementary
to the validate_tradeoff figure.

C-index is recomputed from each fold's predictions pkl by
survot_rank.evidence.manifest.predictions (same helper the official
figure uses). Profile hash check matches the official validate_tradeoff.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from survot_rank.evidence.manifest import predictions as read_predictions

EXPORT_DIR = ROOT / "results/v313_tradeoff_real_20261008/exports_10cohort_5fold_full"
OUT = ROOT / "paper/figures/v313_tradeoff_real_20261008/tradeoff_ten_cohort_full_only"
OUT.mkdir(parents=True, exist_ok=True)

CANCERS = ["BLCA", "BRCA", "COADREAD", "HNSC", "KIRC", "LUAD", "LUSC", "SKCM", "STAD", "UCEC"]


def text_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


rows = []
for cancer in CANCERS:
    cindex, mem, lat, params = [], [], [], None
    for fold in range(5):
        p = EXPORT_DIR / f"exp6_{cancer.lower()}_f{fold}_s3" / "export.json"
        if not p.exists():
            sys.exit(f"missing export: {p}")
        e = json.loads(p.read_text())
        pred_path = ROOT / e["run"]["predictions"]
        if file_sha(pred_path) != e["hashes"]["predictions"]:
            sys.exit(f"hash mismatch in {p}")
        cindex.append(float(read_predictions(pred_path)["cindex"]))
        prof = e["profile"]
        mem.append(prof["peak_allocated_mb"])
        lat.append(prof["latency_ms_median"])
        if params is None:
            params = prof["parameters"]
    rows.append(dict(
        cancer=cancer,
        cindex=float(np.mean(cindex)),
        memory_mib=float(np.mean(mem)),
        latency_ms=float(np.mean(lat)),
        parameters=params,
    ))


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
fig.suptitle("DCT v3.13 Full: per-cohort 5-fold macro average", fontweight="bold")
cmap = plt.get_cmap("tab10")
for i, r in enumerate(rows):
    color = cmap(i % 10)
    for ax, metric, label in zip(
        axes, ("memory_mib", "latency_ms"),
        ("Peak GPU allocated memory (MiB)", "Median forward wall-clock (ms)"),
    ):
        ax.scatter(r[metric], r["cindex"], s=70, color=color, zorder=3)
        ax.annotate(r["cancer"], (r[metric], r["cindex"]),
                    xytext=(5, 5 if i % 2 == 0 else -12),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel(label)
    ax.set_ylabel("5-fold macro C-index")
    ax.grid(ls="--", alpha=0.35)
    ax.margins(0.18)

fig.text(0.5, 0.02,
         "Same protocol and timing settings for every cohort; one point per cohort. "
         "The KIRC/BLCA Direct comparison is in tradeoff_blca_kirc/efficiency_tradeoff.png.",
         ha="center", fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
for ext in ("png", "pdf", "svg"):
    p = OUT / f"efficiency_per_cohort.{ext}"
    fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
plt.close(fig)

prov = dict(
    schema_version=1,
    context=dict(cancers=CANCERS, endpoint="DSS", encoder="UNI2-h", protocol="legacy_val", seed=3),
    note=("Per-cohort 5-fold macro average; same RTX 5090 / 2048 patch / batch 1 / "
          "repeats 30 / warmup 5 measurement protocol. Each (cancer, fold) is the "
          "same 10-cohort Full export that contributes to tradeoff_input_blca_kirc.json "
          "for BLCA/KIRC. C-index recomputed from predictions pkl."),
    points=[dict(cancer=r["cancer"], cindex=r["cindex"], memory_mib=r["memory_mib"],
                 latency_ms=r["latency_ms"], parameters=r["parameters"]) for r in rows],
)
(OUT / "efficiency_per_cohort.json").write_text(json.dumps(prov, indent=2))
print(f"[tradeoff_per_cohort] saved {OUT}")
for r in rows:
    print(f"  {r['cancer']:8s} cindex={r['cindex']:.4f}  mem={r['memory_mib']:.1f} MiB  lat={r['latency_ms']:.1f} ms")
