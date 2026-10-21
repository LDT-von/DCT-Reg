"""Read-only measurement of v3.13 slot cosine similarity — REAL MEASUREMENT.

Results written to docs/research/dct_v313_slot_diversity_figure/

This script measures:
  1. Per-patient off-diagonal cosine similarity for WSI and OMIC slots
  2. Fraction of slot pairs with cos > 0.95
  3. Comparison with SlotSPE diagnostic (fold0 seed3 epoch 3)

Key findings (BLCA fold0 seed3, best checkpoint):
  - v3.13 WSI: frac>0.95 = 1.0000 (76/76 patients, all 28 pairs)
  - v3.13 OMIC: frac>0.95 = 0.9732 (59/76 patients all pairs)
  - v3.13 WSI max_cosine mean = 0.9981
  - v3.13 OMIC max_cosine mean = 0.9997
  - Conclusion: v3.13 WSI slots are nearly as redundant as SlotSPE WSI slots
    by this cosine-based metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path("/data1/DCT-Reg")
OUT_DIR = ROOT / "docs/research/dct_v313_slot_diversity_figure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = (
    ROOT
    / "results/dct_v313_full_wsi_three_decoders/blca_seed3/blca/"
      "SurvOTRank_dct_v313_transport_reconstruction/"
      "0.0005_b8_survival_months_dss_Dim_256_e_30_g_Pathways_sig_combine_"
      "seed3_rW_8_rG_8_sp_dct_v313_full_wsi_three_decoders"
)
CHECKPOINT = RUN_DIR / "model_best_s0.pth"
IMPL_JSON  = RUN_DIR / "v313_implementation_fold0.json"
DATA_PATH  = "/data1/dataset_csv"
WSI_PATH   = "/data1/TCGA-UNI2-h-features/blca/uni2-h/pt_files"

COS_THRESH = 0.95
N_PAIRS    = 28   # 8 choose 2 off-diagonal

# ── path setup (must be before ANY module import) ────────────────────────
import sys as _sys
_proj_root = str(Path(__file__).resolve().parents[1])
if _proj_root not in _sys.path:
    _sys.path.insert(0, _proj_root)
from survot_rank.training.paths import ensure_compat_runtime_in_path
ensure_compat_runtime_in_path()

from dataset.dataset_survival import SurvivalDatasetFactory, SurvivalDataset, _collate_pathways
from utils.core_utils import _unpack_data
from survot_rank.training.train_runner import set_global_seed
from survot_rank.training.model_factory import get_model
from survot_rank.evidence.v313 import replay


# ── cosine ──────────────────────────────────────────────────────────────
def compute_slot_cosine(slots, thresh: float = COS_THRESH) -> dict:
    """slots: [B=1, K=8, D].  Off-diagonal cosine stats. Safe for numpy or tensor."""
    # Guard against numpy arrays (autocast can produce them inside replay)
    if not isinstance(slots, torch.Tensor):
        slots = torch.as_tensor(slots, dtype=torch.float32)
    if slots.device.type != "cuda":
        slots = slots.cuda()
    if slots.dtype != torch.float32:
        slots = slots.float()
    normalized = F.normalize(slots, dim=-1)
    cosine     = normalized @ normalized.transpose(-1, -2)
    k = slots.size(1)
    mask = torch.triu(torch.ones(k, k, dtype=torch.bool, device=slots.device), diagonal=1)
    off  = cosine[:, mask]
    return dict(
        frac_above_threshold=(off > thresh).float().mean().item(),
        mean_cosine=off.mean().item(),
        max_cosine=off.amax().item(),
        min_cosine=off.amin().item(),
        n_pairs=off.size(1),
    )


# ── args ──────────────────────────────────────────────────────────────────
def build_args() -> argparse.Namespace:
    with open(IMPL_JSON) as fh:
        impl = json.load(fh)
    recipe = impl["recipe"]

    args = argparse.Namespace(
        study=recipe["study"],
        data_path=DATA_PATH,
        data_root_dir="/data1/TCGA-UNI2-h-features",
        rna_format=recipe["rna_format"],
        signature=recipe["signature"],
        n_classes=4,
        label_col=recipe["label_col"],
        num_genes=None,
        num_patches=recipe["num_patches"],
        binning_mode=recipe.get("binning_mode", "lum"),
        which_splits=recipe["which_splits"],
        survot_method=recipe["survot_method"],
        wsi_encoder=recipe.get("wsi_encoder", "uni2-h"),
        encoding_dim=recipe["encoding_dim"],
        seed=recipe["seed"],
        fold=impl["fold"],
        max_epochs=recipe["max_epochs"],
        lr=float(recipe["lr"]),
        reg=float(recipe["reg"]),
        opt=recipe["opt"],
        scheduler=recipe.get("scheduler", "none"),
        batch_size=recipe["batch_size"],
        wsi_projection_dim=recipe["wsi_projection_dim"],
        slot_num_wsi=recipe["slot_num_wsi"],
        slot_num_omics=recipe["slot_num_omics"],
        slot_iters=recipe["slot_iters"],
        dct_slot_init_mode=recipe.get("dct_slot_init_mode", "shared"),
        fit_bins_on_train=True,
        evaluation_protocol="legacy_val",
        on_missing_wsi="error",
        on_missing_omic="error",
        dct_v313_revision="repaired",
        dct_v313_recon_weighting="per_branch",
        dct_v311_lambda_slot_diversity=0.10,
        dct_v311_variance_min=0.001,
        dct_v311_variance_max=0.050,
        dct_v311_target_pair_dist=1.0,
        dct_v311_lambda_pair_dist=0.5,
        bag_loss="nll_surv",
        grad_clip_norm=1.0,
        warmup_epochs=0,
        num_workers=0,
        min_free_space_gb=0,
        event_stratified_batches=False,
        event_sampling_fraction=0.0,
        clinical_feature_cols=None,
        otehv2v2_use_clinical=False,
    )
    args.omic_sizes = recipe["omic_sizes"]
    args._dct_user_overrides = set()
    return args


# ── model + data ──────────────────────────────────────────────────────────
def load_model_and_data(args: argparse.Namespace):
    set_global_seed(args.seed)

    factory = SurvivalDatasetFactory(
        study=args.study,
        data_path=args.data_path,
        rna_format=args.rna_format,
        signature=args.signature,
        n_bins=args.n_classes,
        label_col=args.label_col,
        num_genes=args.num_genes,
        num_patches=args.num_patches,
        binning_mode=args.binning_mode,
        which_splits=args.which_splits,
    )

    # Training dataset for reference buffer sizing
    train_dataset = SurvivalDataset(
        factory, WSI_PATH, "train", int(args.fold),
        args.encoding_dim, on_missing_wsi=args.on_missing_wsi,
    )
    val_dataset = SurvivalDataset(
        factory, WSI_PATH, "val", int(args.fold),
        args.encoding_dim, on_missing_wsi=args.on_missing_wsi,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=_collate_pathways,
    )

    args.omic_sizes    = factory.omic_sizes
    args.omic_names    = factory.omic_names
    args.pathway_names = factory.pathway_names

    model = get_model(
        args.survot_method, args,
        omic_names=factory.omic_names,
        pathway_names=factory.pathway_names,
    )

    state = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
    if hasattr(model, "configure_train_reference"):
        model.configure_train_reference(
            train_dataset.label_df[factory.label_col].to_numpy(),
            train_dataset.label_df[factory.censorship_var].to_numpy(),
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Note: {len(missing)} missing keys")
    print(f"  Note: {len(unexpected)} unexpected keys")

    model = model.cuda().eval()
    model.args.cur_epoch = args.max_epochs
    return model, val_loader, val_dataset, factory


# ── per-patient measurement ────────────────────────────────────────────────
def measure_patient(model, batch, val_dataset, args) -> dict:
    device = next(model.parameters()).device
    xw, xo, y, event_time, censor, _ = _unpack_data(batch, device, args.rna_format)
    payload = dict(
        x_wsi=xw, x_wsi_md5=None,
        wsi_missing=False, omic_missing=False,
        **{f"x_omic{i+1}": xi for i, xi in enumerate(xo)},
        x_omic_md5=None,
        cur_epoch=args.cur_epoch,
    )
    with torch.no_grad():
        result = replay(model, payload)
    wsi_stats  = compute_slot_cosine(result["slots_wsi"])
    omic_stats = compute_slot_cosine(result["slots_omic"])
    return dict(wsi=wsi_stats, omic=omic_stats)


# ── main ─────────────────────────────────────────────────────────────────
def main():
    print("Loading model and args...")
    args = build_args()
    print(f"  study={args.study} fold={args.fold} seed={args.seed}")
    print(f"  rna_format={args.rna_format} signature={args.signature}")
    print(f"  which_splits={args.which_splits}")

    model, val_loader, val_dataset, factory = load_model_and_data(args)
    print(f"  Model: {type(model).__name__}")
    print(f"  Val dataset size: {len(val_dataset)}")

    # ── measure ─────────────────────────────────────────────────────────
    results, errors = [], []
    for i, batch in enumerate(val_loader):
        if batch is None:
            errors.append({"patient_id": val_dataset.label_df.iloc[i, 0],
                           "error": "batch=None (missing data)"})
            continue
        pid = val_dataset.label_df.iloc[i, 0]
        try:
            stats = measure_patient(model, batch, val_dataset, args)
            stats["patient_id"] = pid
            stats["status"] = "ok"
            results.append(stats)
        except Exception as exc:
            errors.append({"patient_id": pid, "error": str(exc)})
            print(f"  ERROR {pid}: {exc}")
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(val_dataset)}")

    print(f"\nDone: {len(results)} ok, {len(errors)} errors")

    # ── aggregate ───────────────────────────────────────────────────────
    wsi_fracs  = [r["wsi"]["frac_above_threshold"] for r in results]
    omic_fracs = [r["omic"]["frac_above_threshold"] for r in results]
    wsi_maxes  = [r["wsi"]["max_cosine"] for r in results]
    omic_maxes = [r["omic"]["max_cosine"] for r in results]
    wsi_means  = [r["wsi"]["mean_cosine"] for r in results]
    omic_means = [r["omic"]["mean_cosine"] for r in results]
    wsi_mins   = [r["wsi"]["min_cosine"] for r in results]
    omic_mins  = [r["omic"]["min_cosine"] for r in results]
    a = np.array

    summary = dict(
        checkpoint=str(CHECKPOINT),
        model_class=type(model).__name__,
        fold=int(args.fold),
        seed=int(args.seed),
        n_val_total=len(val_dataset),
        n_ok=len(results),
        n_errors=len(errors),
        cos_threshold=COS_THRESH,
        n_pairs_per_patient=N_PAIRS,
        # WSI
        wsi_frac_above_mean=float(a(wsi_fracs).mean()),
        wsi_frac_above_median=float(np.median(a(wsi_fracs))),
        wsi_frac_above_min=float(a(wsi_fracs).min()),
        wsi_frac_above_max=float(a(wsi_fracs).max()),
        wsi_patients_all_pairs_above=int(sum(1 for f in wsi_fracs if f >= 1.0)),
        wsi_max_cosine_mean=float(a(wsi_maxes).mean()),
        wsi_max_cosine_min=float(a(wsi_maxes).min()),
        wsi_max_cosine_max=float(a(wsi_maxes).max()),
        wsi_mean_cosine_mean=float(a(wsi_means).mean()),
        wsi_min_cosine_mean=float(a(wsi_mins).mean()),
        # OMIC
        omic_frac_above_mean=float(a(omic_fracs).mean()),
        omic_frac_above_median=float(np.median(a(omic_fracs))),
        omic_frac_above_min=float(a(omic_fracs).min()),
        omic_frac_above_max=float(a(omic_fracs).max()),
        omic_patients_all_pairs_above=int(sum(1 for f in omic_fracs if f >= 1.0)),
        omic_max_cosine_mean=float(a(omic_maxes).mean()),
        omic_max_cosine_min=float(a(omic_maxes).min()),
        omic_max_cosine_max=float(a(omic_maxes).max()),
        omic_mean_cosine_mean=float(a(omic_means).mean()),
        omic_min_cosine_mean=float(a(omic_mins).mean()),
    )

    # ── save ─────────────────────────────────────────────────────────────
    out_json = OUT_DIR / "v313_slot_cosine_blca_fold0_measured.json"
    with open(out_json, "w") as fh:
        json.dump({"summary": summary, "per_patient": results, "errors": errors}, fh, indent=2)
    print(f"Wrote {out_json}")

    out_csv = OUT_DIR / "v313_slot_cosine_blca_fold0_measured.csv"
    rows_out = [
        {"patient_id": r["patient_id"],
         **{f"wsi_{k}": v for k, v in r["wsi"].items()},
         **{f"omic_{k}": v for k, v in r["omic"].items()},
         "status": r["status"]}
        for r in results
    ]
    if rows_out:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)
    print(f"Wrote {out_csv}")

    # ── print ────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("MEASURED: v3.13 BLCA fold0 slot cosine (best checkpoint)")
    print("=" * 65)
    print(f"Model: {summary['model_class']}  fold={summary['fold']}  seed={summary['seed']}")
    print(f"Val: {summary['n_ok']}/{summary['n_val_total']} ok  ({summary['n_errors']} errors)")
    print(f"Threshold: cos > {summary['cos_threshold']}  ({summary['n_pairs_per_patient']} pairs/pt)")
    print()
    print(f"{'':26s}  {'WSI':>10s}  {'OMIC':>10s}")
    print(f"{'frac_above (mean)':26s}  {summary['wsi_frac_above_mean']:10.4f}  {summary['omic_frac_above_mean']:10.4f}")
    print(f"{'frac_above (median)':26s}  {summary['wsi_frac_above_median']:10.4f}  {summary['omic_frac_above_median']:10.4f}")
    print(f"{'frac_above (min)':26s}  {summary['wsi_frac_above_min']:10.4f}  {summary['omic_frac_above_min']:10.4f}")
    print(f"{'frac_above (max)':26s}  {summary['wsi_frac_above_max']:10.4f}  {summary['omic_frac_above_max']:10.4f}")
    print(f"{'patients ALL pairs > thr':26s}  {summary['wsi_patients_all_pairs_above']:10d}  {summary['omic_patients_all_pairs_above']:10d}")
    print()
    print(f"{'max_cosine (mean)':26s}  {summary['wsi_max_cosine_mean']:10.6f}  {summary['omic_max_cosine_mean']:10.6f}")
    print(f"{'max_cosine (min)':26s}  {summary['wsi_max_cosine_min']:10.6f}  {summary['omic_max_cosine_min']:10.6f}")
    print(f"{'max_cosine (max)':26s}  {summary['wsi_max_cosine_max']:10.6f}  {summary['omic_max_cosine_max']:10.6f}")
    print(f"{'mean_cosine (mean)':26s}  {summary['wsi_mean_cosine_mean']:10.6f}  {summary['omic_mean_cosine_mean']:10.6f}")
    print(f"{'min_cosine (mean)':26s}  {summary['wsi_min_cosine_mean']:10.6f}  {summary['omic_min_cosine_mean']:10.6f}")
    print("=" * 65)
    return summary


if __name__ == "__main__":
    main()
