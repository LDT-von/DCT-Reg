#!/usr/bin/env python3
"""Redesigned dose-monotonicity audit for v3.10 BLCA.

Goal: replace the 11-alpha strict monotonicity sweep with a more
interpretable end-to-end design.

Design:
  - 2 alpha points: {0.0 (factual), 1.0 (full anchor)}.
  - For each fold, run two sweeps:
      * REAL: use the trained risk anchors.
      * RANDOM: shuffle the slot dimension of both anchors
        (preserving marginals but destroying semantic anchor identity).
  - Compare end-to-end directional rates:
      * end_to_end_high_rate = P(risk[α=1, HIGH anchor] > risk[α=0])
      * end_to_end_low_rate  = P(risk[α=1, LOW  anchor] < risk[α=0])
  - Report info_gap = real_rate - random_rate (signal beyond chance).

No re-training required; reuses v3.10 BLCA best checkpoints under
results/dct_v3.10/robust/final_50ep_old/blca/...

Usage:
  python scripts/redesign_audit_blca.py --fold 0 --gpu 0
  python scripts/redesign_audit_blca.py --fold all --gpu 0
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survot_rank.cli import add_project_paths  # noqa: E402

add_project_paths()

from survot_rank.config import (  # noqa: E402
    apply_overrides,
    config_to_argv,
    load_config,
)
from survot_rank.training.extended_args import process_args_extended  # noqa: E402
from survot_rank.training.model_factory import get_model  # noqa: E402

try:
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (  # noqa: E402
        SurvivalDatasetFactory,
    )
except ImportError:
    SurvivalDatasetFactory = None  # type: ignore

try:
    from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (  # noqa: E402
        _process_data_and_forward,
    )
except ImportError:
    _process_data_and_forward = None  # type: ignore

try:
    from survot_rank.training.sparse_event import get_split  # noqa: E402
except ImportError:
    get_split = None  # type: ignore
if get_split is None:
    from survot_rank.training.train_runner import get_split  # noqa: E402


# ---------------------------------------------------------------------------
# Loaders (mirror audit_dct_reg.py: keep aligned with cmd_sweep)
# ---------------------------------------------------------------------------


def _load_parsed_args(args):
    config = apply_overrides(load_config(args.config), args.set or [])
    parsed = process_args_extended(config_to_argv(config))
    return parsed


def _load_model_and_loader(args, parsed, fold: int):
    factory = SurvivalDatasetFactory(
        study=parsed.study,
        data_path=parsed.data_path,
        rna_format=parsed.rna_format,
        signature=parsed.signature,
        n_bins=parsed.n_classes,
        label_col=parsed.label_col,
        num_genes=parsed.num_genes,
        num_patches=parsed.num_patches,
        clinical_feature_cols=(
            [c.strip() for c in parsed.clinical_feature_cols.split(",") if c.strip()]
            if getattr(parsed, "clinical_feature_cols", None)
            else None
        ),
        binning_mode=getattr(parsed, "binning_mode", "global_qcut"),
    )
    if parsed.rna_format in ("Pathways", "RNASeq", "GeneEmbedding"):
        rna_cases = set(factory.gene_data_df.columns)
        factory.clinical_df = factory.clinical_df[
            factory.clinical_df["case id"].isin(rna_cases)
        ].reset_index(drop=True)

    train_data, val_data, _, val_loader = get_split(parsed, factory, fold)
    parsed.omic_sizes = factory.omic_sizes
    parsed.omic_names = factory.omic_names
    parsed.pathway_names = getattr(factory, "pathway_names", None)
    if parsed.rna_format == "RNASeq":
        omics_input_dim = (
            factory.num_genes if factory.num_genes is not None else factory.omic_sizes
        )
    elif parsed.rna_format == "GeneEmbedding":
        omics_input_dim = 768
    else:
        omics_input_dim = None
    model = get_model(
        method=parsed.survot_method,
        args=parsed,
        omic_input_dim=omics_input_dim,
        omic_names=parsed.omic_names,
        pathway_names=parsed.pathway_names,
    )
    model.configure_train_reference(
        train_data.label_df[factory.label_col].to_numpy(),
        train_data.label_df[factory.censorship_var].to_numpy(),
    )

    # Load checkpoint with shape filtering (matches audit_dct_reg._load_model_and_loader)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_state_keys = set(model.state_dict().keys())
    filtered_state_dict = {}
    skipped = []
    for key, value in state_dict.items():
        if key in model_state_keys:
            model_shape = model.state_dict()[key].shape
            checkpoint_shape = value.shape
            if model_shape == checkpoint_shape:
                filtered_state_dict[key] = value
            else:
                skipped.append(f"{key}: {checkpoint_shape} -> {model_shape}")
        else:
            filtered_state_dict[key] = value
    if skipped:
        print(f"[fold {fold}] skipped {len(skipped)} size-mismatched (e.g. {skipped[0] if skipped else ''})")
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, val_loader, val_data, factory, device


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------


def _interpolate_cost(factual_costs, anchor_costs, alpha, seen_mask):
    bsz = factual_costs.size(0)
    expanded = anchor_costs.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
    seen = seen_mask.view(1, -1, 1, 1, 1)
    expanded = torch.where(seen, expanded, factual_costs)
    return (1.0 - alpha) * factual_costs + alpha * expanded


def _risk_for_anchor(model, parsed, val_loader, anchor_idx, device, alpha_list):
    """Collect per-case risk trajectory at each alpha for one anchor."""
    per_case: list[list[float]] = []
    for data in val_loader:
        out, _, _, _ = _process_data_and_forward(parsed, model, data, device, test=True)
        _, _ = out
        explanations = model.last_explanations
        if explanations is None:
            continue
        factual_costs = getattr(model, "_last_factual_costs", None)
        if factual_costs is None:
            continue
        rows = model._last_factual_rows
        cols = model._last_factual_cols
        slots_wsi = model._last_slots_wsi
        slots_omic = model._last_slots_omic
        epoch = int(getattr(parsed, "cur_epoch", 0))
        for case_idx in range(factual_costs.size(0)):
            risks: list[float] = []
            for alpha in alpha_list:
                costs = _interpolate_cost(
                    factual_costs[case_idx:case_idx + 1],
                    model.risk_anchor_costs[:, anchor_idx],
                    alpha,
                    model.risk_anchor_seen[:, anchor_idx],
                )
                plan, _ = model._plans_from_cost_tensor(
                    costs,
                    rows[case_idx:case_idx + 1],
                    cols[case_idx:case_idx + 1],
                    epoch,
                )
                logits, _ = model._encode_logits_from_plans(
                    slots_wsi[case_idx:case_idx + 1],
                    slots_omic[case_idx:case_idx + 1],
                    plan,
                )
                r = float(model._risk(logits).detach().cpu().numpy()[0])
                risks.append(r)
            per_case.append(risks)
    return per_case


def _shuffle_slot_dim(anchor: torch.Tensor) -> torch.Tensor:
    """Permute the slot-dim of an anchor, preserving tensor shape & marginals.

    Anchor shape (per audit_dct_reg._interpolate_cost) is
    [num_stages, 2, num_slots_wsi, slot_dim_wsi, slot_dim_omics].
    We permute axis=2 (num_slots_wsi) independently per stage.
    """
    out = anchor.clone()
    n_slots = out.size(2)
    device = out.device
    for stage in range(out.size(0)):
        perm = torch.randperm(n_slots, device=device)
        out[stage] = out[stage].index_select(2, perm)
    return out


def run_fold(fold: int, args, alpha_list=(0.0, 1.0)) -> dict:
    parsed = _load_parsed_args(args)
    parsed.k_start = fold
    parsed.k_end = fold + 1
    parsed.cur_fold = fold
    parsed.cur_epoch = int(getattr(args, "epoch", 0))
    parsed.num_workers = 4

    model, val_loader, val_data, factory, device = _load_model_and_loader(
        args, parsed, fold
    )

    # Sanity: anchor shape
    anchor_shape = tuple(model.risk_anchor_costs.shape)
    print(f"[fold {fold}] anchor shape: {anchor_shape}")
    print(f"[fold {fold}] LOW_RISK idx: {model._LOW_RISK}, HIGH_RISK idx: {model._HIGH_RISK}")

    # ---- REAL sweep ----
    real_low = _risk_for_anchor(model, parsed, val_loader, model._LOW_RISK, device, alpha_list)
    real_high = _risk_for_anchor(model, parsed, val_loader, model._HIGH_RISK, device, alpha_list)

    # ---- RANDOM sweep (shuffled slot dim) ----
    saved_low = model.risk_anchor_costs[:, model._LOW_RISK].clone()
    saved_high = model.risk_anchor_costs[:, model._HIGH_RISK].clone()
    torch.manual_seed(0)  # reproducible shuffle
    model.risk_anchor_costs[:, model._LOW_RISK] = _shuffle_slot_dim(saved_low)
    model.risk_anchor_costs[:, model._HIGH_RISK] = _shuffle_slot_dim(saved_high)
    rand_low = _risk_for_anchor(model, parsed, val_loader, model._LOW_RISK, device, alpha_list)
    rand_high = _risk_for_anchor(model, parsed, val_loader, model._HIGH_RISK, device, alpha_list)
    # restore
    model.risk_anchor_costs[:, model._LOW_RISK] = saved_low
    model.risk_anchor_costs[:, model._HIGH_RISK] = saved_high

    return {
        "fold": fold,
        "alphas": list(alpha_list),
        "real_low": real_low,
        "real_high": real_high,
        "rand_low": rand_low,
        "rand_high": rand_high,
        "anchor_shape": list(anchor_shape),
    }


def _to_numpy_per_case(per_case) -> np.ndarray:
    return np.asarray(per_case, dtype=np.float64)


def compute_end_to_end(per_case, alpha_list) -> dict[str, float]:
    arr = _to_numpy_per_case(per_case)
    if arr.size == 0:
        return {"n_cases": 0}
    a0 = arr[:, 0]
    a1 = arr[:, -1]
    delta = a1 - a0
    return {
        "n_cases": int(arr.shape[0]),
        "delta_mean": float(delta.mean()),
        "delta_std": float(delta.std()),
        "delta_abs_mean": float(np.abs(delta).mean()),
        "rate_up": float((delta > 0).mean()),
        "rate_down": float((delta < 0).mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fold", required=True, help="0..4 or 'all'")
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output-dir", default="results/audit_blca_redesign/blca")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    folds = list(range(5)) if args.fold == "all" else [int(args.fold)]

    summary: dict[int, dict] = {}
    for fold in folds:
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        out = run_fold(fold, args)
        elapsed = time.time() - t0

        pkl_path = fold_dir / f"redesigned_sweep_fold{fold}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(out, f)

        metrics = {
            "fold": fold,
            "checkpoint": str(args.checkpoint),
            "alphas": out["alphas"],
            "anchor_shape": out["anchor_shape"],
            "elapsed_sec": round(elapsed, 1),
            "real_low_end_to_end": compute_end_to_end(out["real_low"], out["alphas"]),
            "real_high_end_to_end": compute_end_to_end(out["real_high"], out["alphas"]),
            "rand_low_end_to_end": compute_end_to_end(out["rand_low"], out["alphas"]),
            "rand_high_end_to_end": compute_end_to_end(out["rand_high"], out["alphas"]),
        }
        # info_gap
        m = metrics
        m["info_gap_high_rate"] = (
            m["real_high_end_to_end"]["rate_up"]
            - m["rand_high_end_to_end"]["rate_up"]
        )
        m["info_gap_low_rate"] = (
            m["real_low_end_to_end"]["rate_down"]
            - m["rand_low_end_to_end"]["rate_down"]
        )
        m["info_gap_high_delta_abs"] = (
            m["real_high_end_to_end"]["delta_abs_mean"]
            - m["rand_high_end_to_end"]["delta_abs_mean"]
        )
        m["info_gap_low_delta_abs"] = (
            m["real_low_end_to_end"]["delta_abs_mean"]
            - m["rand_low_end_to_end"]["delta_abs_mean"]
        )
        json_path = fold_dir / f"redesigned_metrics_fold{fold}.json"
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[fold {fold}] done in {elapsed:.1f}s, saved {pkl_path.name}")
        print(
            f"  real_high rate_up={m['real_high_end_to_end']['rate_up']:.4f}, "
            f"rand_high rate_up={m['rand_high_end_to_end']['rate_up']:.4f}, "
            f"info_gap_high_rate={m['info_gap_high_rate']:.4f}"
        )
        print(
            f"  real_low  rate_down={m['real_low_end_to_end']['rate_down']:.4f}, "
            f"rand_low  rate_down={m['rand_low_end_to_end']['rate_down']:.4f}, "
            f"info_gap_low_rate={m['info_gap_low_rate']:.4f}"
        )
        summary[fold] = metrics

    # 5-fold summary
    if len(folds) == 5:
        real_h_up = [summary[f]["real_high_end_to_end"]["rate_up"] for f in folds]
        rand_h_up = [summary[f]["rand_high_end_to_end"]["rate_up"] for f in folds]
        real_l_dn = [summary[f]["real_low_end_to_end"]["rate_down"] for f in folds]
        rand_l_dn = [summary[f]["rand_low_end_to_end"]["rate_down"] for f in folds]
        info_h = [summary[f]["info_gap_high_rate"] for f in folds]
        info_l = [summary[f]["info_gap_low_rate"] for f in folds]
        summary_5fold = {
            "real_high_rate_up_mean": float(np.mean(real_h_up)),
            "rand_high_rate_up_mean": float(np.mean(rand_h_up)),
            "real_low_rate_down_mean": float(np.mean(real_l_dn)),
            "rand_low_rate_down_mean": float(np.mean(rand_l_dn)),
            "info_gap_high_mean": float(np.mean(info_h)),
            "info_gap_high_std": float(np.std(info_h)),
            "info_gap_low_mean": float(np.mean(info_l)),
            "info_gap_low_std": float(np.std(info_l)),
        }
        with open(output_dir / "summary_5fold.json", "w") as f:
            json.dump(summary_5fold, f, indent=2)
        print("\n=== 5-fold summary ===")
        print(json.dumps(summary_5fold, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
