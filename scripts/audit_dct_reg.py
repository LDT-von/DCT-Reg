#!/usr/bin/env python3
"""Held-out mechanism audit for frozen DCT v3.10 / DCT-Reg.

Loads a per-fold checkpoint, re-runs a labeled held-out forward pass, and
writes per-case risk deltas plus direction, dose and coupling-reconfiguration
metrics.  Use ``python scripts/audit_dct_reg.py audit --help`` for one fold or
``python scripts/audit_dct_reg.py sweep --help`` for a batch audit.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
from survot_rank.training.evidence import finalize_fold_evidence  # noqa: E402

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
# Audit metric helpers — pure numpy / torch, no model-side coupling.
# ---------------------------------------------------------------------------


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@torch.no_grad()
def apply_anchor_mode(model, mode: str) -> None:
    """Apply an audit-only anchor identity control after checkpoint loading."""

    if mode == "normal":
        return
    if mode != "swapped":
        raise ValueError("anchor mode must be 'normal' or 'swapped'")
    order = torch.tensor(
        [model._HIGH_RISK, model._LOW_RISK],
        device=model.risk_anchor_costs.device,
        dtype=torch.long,
    )
    model.risk_anchor_costs.copy_(model.risk_anchor_costs.index_select(1, order))
    model.risk_anchor_seen.copy_(model.risk_anchor_seen.index_select(1, order))


def direction_consistency(
    factual_risk: np.ndarray,
    low_risk: np.ndarray,
    high_risk: np.ndarray,
    event_time: np.ndarray,
    censorship: np.ndarray,
    observed_quantile: float = 0.40,
) -> dict[str, float]:
    """Fraction of cases whose counterfactual delta agrees with their label.

    A case is "high-risk-labelled" if its observed event time lies below the
    ``observed_quantile`` of all observed events. A case is "low-risk-labelled"
    if its observed event time lies above the upper quantile or it is censored
    past the upper edge. For high-labelled cases we expect
    ``high_risk - factual_risk > 0``; for low-labelled cases we expect
    ``low_risk - factual_risk < 0``.

    Returns counts and rates; ``chance_gap`` is ``correct_rate - 0.5`` and is
    therefore in [-0.5, 0.5].
    """
    observed = censorship < 0.5
    observed_times = event_time[observed]
    if observed_times.size < 8:
        return {
            "high_labelled_count": 0,
            "low_labelled_count": 0,
            "correct_rate": float("nan"),
            "chance_gap": float("nan"),
        }
    upper = float(np.quantile(observed_times, 1.0 - observed_quantile))
    lower = float(np.quantile(observed_times, observed_quantile))
    high_mask = observed & (event_time <= lower)
    low_mask = (event_time > upper) | ((censorship >= 0.5) & (event_time >= upper))
    high_delta = high_risk - factual_risk
    low_delta = low_risk - factual_risk
    high_correct = int((high_delta[high_mask] > 0).sum()) if high_mask.any() else 0
    low_correct = int((low_delta[low_mask] < 0).sum()) if low_mask.any() else 0
    high_total = int(high_mask.sum())
    low_total = int(low_mask.sum())
    labelled = high_total + low_total
    correct = high_correct + low_correct
    correct_rate = (correct / labelled) if labelled else float("nan")
    return {
        "high_labelled_count": high_total,
        "low_labelled_count": low_total,
        "high_correct": high_correct,
        "low_correct": low_correct,
        "correct_rate": float(correct_rate),
        "chance_gap": float(correct_rate - 0.5),
    }


def bidirectional_direction_consistency(
    factual_risk: np.ndarray,
    low_risk: np.ndarray,
    high_risk: np.ndarray,
) -> dict[str, float]:
    """Direction response over every held-out case, matching the train objective."""

    if factual_risk.size == 0:
        return {
            "n_cases": 0,
            "high_correct_rate": float("nan"),
            "low_correct_rate": float("nan"),
            "both_correct_rate": float("nan"),
            "side_correct_rate": float("nan"),
        }
    high_correct = high_risk > factual_risk
    low_correct = low_risk < factual_risk
    return {
        "n_cases": int(factual_risk.size),
        "high_correct_rate": float(high_correct.mean()),
        "low_correct_rate": float(low_correct.mean()),
        "both_correct_rate": float((high_correct & low_correct).mean()),
        "side_correct_rate": float(
            (high_correct.sum() + low_correct.sum()) / (2 * factual_risk.size)
        ),
    }


def dose_monotonicity(
    sweep_alphas: np.ndarray,
    sweep_risks: np.ndarray,
    *,
    increasing: bool = True,
) -> dict[str, float]:
    """Rate at which increasing ``alpha`` monotonically raises risk toward high anchor.

    ``sweep_alphas`` has shape ``[n_alphas]`` and ``sweep_risks`` has shape
    ``[n_cases, n_alphas]``. Each case is expected to satisfy
    ``risk[alpha]`` increasing with ``alpha``; the reported rate is the
    fraction of cases for which all consecutive pairs satisfy this.
    """
    if sweep_alphas.size < 2:
        return {"monotone_rate": float("nan"), "n_pairs": 0}
    deltas = np.diff(sweep_risks, axis=1)
    expected_sign = np.sign(np.diff(sweep_alphas)).mean()
    if not increasing:
        expected_sign *= -1.0
    if expected_sign > 0:
        correct = (deltas > 0).all(axis=1)
    elif expected_sign < 0:
        correct = (deltas < 0).all(axis=1)
    else:
        return {"monotone_rate": float("nan"), "n_pairs": 0}
    return {
        "monotone_rate": float(correct.mean()),
        "n_cases": int(sweep_risks.shape[0]),
        "n_pairs": int(deltas.shape[1]),
    }


def reconfiguration_magnitude(
    factual_plans: list[torch.Tensor],
    intervened_plans: list[torch.Tensor],
    margin: float = 0.02,
) -> dict[str, float]:
    """Mean total-variation distance between factual and intervened plans.

    A meaningful re-solve must move the coupling by more than ``margin``;
    otherwise the intervention degenerated to marginal projection without
    geometric change.
    """
    distances = []
    for stage_factual, stage_intervened in zip(factual_plans, intervened_plans):
        diff = (stage_factual - stage_intervened).abs().sum(dim=(-1, -2)) * 0.5
        distances.append(diff.detach().cpu().numpy())
    if not distances:
        return {"mean_tv": float("nan"), "above_margin_rate": float("nan")}
    stacked = np.stack(distances, axis=0)  # [n_stages, n_cases]
    mean_tv = float(stacked.mean())
    return {
        "mean_tv": mean_tv,
        "above_margin_rate": float((stacked > margin).mean()),
        "per_stage_mean_tv": stacked.mean(axis=1).tolist(),
    }


def plan_total_variation(
    factual_plans: np.ndarray,
    low_plans: np.ndarray,
    high_plans: np.ndarray,
) -> dict[str, object]:
    """Compute true coupling TV for [case, stage, geometry, row, col] plans."""

    if factual_plans.size == 0:
        return {
            "mean_tv_low": float("nan"),
            "mean_tv_high": float("nan"),
            "mean_tv": float("nan"),
            "per_case_tv_low": [],
            "per_case_tv_high": [],
        }
    if factual_plans.shape != low_plans.shape or factual_plans.shape != high_plans.shape:
        raise ValueError("factual/low/high plan tensors must have identical shapes")
    low_tv = 0.5 * np.abs(low_plans - factual_plans).sum(axis=(-1, -2))
    high_tv = 0.5 * np.abs(high_plans - factual_plans).sum(axis=(-1, -2))
    per_case_low = low_tv.mean(axis=(1, 2))
    per_case_high = high_tv.mean(axis=(1, 2))
    return {
        "mean_tv_low": float(low_tv.mean()),
        "mean_tv_high": float(high_tv.mean()),
        "mean_tv": float(0.5 * (low_tv.mean() + high_tv.mean())),
        "per_stage_geometry_tv_low": low_tv.mean(axis=0).tolist(),
        "per_stage_geometry_tv_high": high_tv.mean(axis=0).tolist(),
        "per_case_tv_low": per_case_low.tolist(),
        "per_case_tv_high": per_case_high.tolist(),
    }


# ---------------------------------------------------------------------------
# Forward + audit capture
# ---------------------------------------------------------------------------


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
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)
    apply_anchor_mode(model, getattr(args, "anchor_mode", "normal"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, val_loader, val_data, factory


def _run_alpha_sweep(
    model,
    parsed,
    val_loader,
    alphas: Iterable[float],
) -> dict[str, np.ndarray]:
    """Run one extra sweep per alpha value to capture dose_monotonicity data.

    The base DCT ``last_explanations`` only stores alpha=1 (low/high). For
    the monotonicity test we replay the full intervention chain at each
    requested alpha using the cached factual plans and risk anchors. The
    cached anchor buffers live on the model and are train-fold statistics.
    """
    alpha_list = list(alphas)
    device = next(model.parameters()).device
    high_risks_per_case: list[list[float]] = []
    low_risks_per_case: list[list[float]] = []
    sweep_alphas = np.array(alpha_list, dtype=np.float64)
    case_offset = 0
    for batch_idx, data in enumerate(val_loader):
        out, _, _, _ = _process_data_and_forward(
            parsed, model, data, device, test=True
        )
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
            high_for_case: list[float] = []
            low_for_case: list[float] = []
            for alpha in alpha_list:
                side_risks = []
                for risk_index in (model._HIGH_RISK, model._LOW_RISK):
                    intervened_costs = _interpolate_cost(
                        factual_costs[case_idx:case_idx + 1],
                        model.risk_anchor_costs[:, risk_index],
                        alpha,
                        model.risk_anchor_seen[:, risk_index],
                    )
                    plan, _ = model._plans_from_cost_tensor(
                        intervened_costs,
                        rows[case_idx:case_idx + 1],
                        cols[case_idx:case_idx + 1],
                        epoch,
                    )
                    logits, _ = model._encode_logits_from_plans(
                        slots_wsi[case_idx:case_idx + 1],
                        slots_omic[case_idx:case_idx + 1],
                        plan,
                    )
                    side_risks.append(float(model._risk(logits).detach().cpu().numpy()[0]))
                high_for_case.append(side_risks[0])
                low_for_case.append(side_risks[1])
            high_risks_per_case.append(high_for_case)
            low_risks_per_case.append(low_for_case)
        case_offset += factual_costs.size(0)
    if not high_risks_per_case:
        empty = np.empty((0, len(alpha_list)))
        return {
            "sweep_alphas": sweep_alphas,
            "sweep_risks": empty,
            "high_sweep_risks": empty,
            "low_sweep_risks": empty.copy(),
        }
    high = np.asarray(high_risks_per_case, dtype=np.float64)
    low = np.asarray(low_risks_per_case, dtype=np.float64)
    return {
        "sweep_alphas": sweep_alphas,
        # Backward-compatible alias for the historical high-risk-only sweep.
        "sweep_risks": high,
        "high_sweep_risks": high,
        "low_sweep_risks": low,
    }


def _interpolate_cost(
    factual_costs: torch.Tensor,
    anchor_costs: torch.Tensor,
    alpha: float,
    seen_mask: torch.Tensor,
) -> torch.Tensor:
    bsz = factual_costs.size(0)
    expanded = anchor_costs.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
    seen = seen_mask.view(1, -1, 1, 1, 1)
    expanded = torch.where(seen, expanded, factual_costs)
    return (1.0 - alpha) * factual_costs + alpha * expanded


def _stacked_to_nested(plans: torch.Tensor):
    return [
        tuple(plans[:, stage_idx, geometry_idx] for geometry_idx in range(plans.size(2)))
        for stage_idx in range(plans.size(1))
    ]


@torch.no_grad()
def controlled_plan_risk(model, control: str, seed: int = 1):
    """Decode a feasible uniform or shuffled factual-plan control."""

    if control not in {"uniform", "shuffled"}:
        raise ValueError("control must be 'uniform' or 'shuffled'")
    factual = model.last_explanations["factual_transport_plans"]
    rows = model._last_factual_rows
    cols = model._last_factual_cols
    if control == "uniform":
        independent = rows.unsqueeze(2).unsqueeze(-1) * cols.unsqueeze(2).unsqueeze(-2)
        controlled = independent.expand(-1, -1, factual.size(2), -1, -1).clone()
    else:
        generator = torch.Generator(device=factual.device)
        generator.manual_seed(int(seed))
        row_order = torch.randperm(factual.size(-2), generator=generator, device=factual.device)
        col_order = torch.randperm(factual.size(-1), generator=generator, device=factual.device)
        controlled = factual.index_select(-2, row_order).index_select(-1, col_order)
        projected = []
        for stage_idx in range(controlled.size(1)):
            stage = []
            for geometry_idx in range(controlled.size(2)):
                stage.append(
                    model._project_coupling(
                        controlled[:, stage_idx, geometry_idx],
                        rows[:, stage_idx],
                        cols[:, stage_idx],
                    )
                )
            projected.append(torch.stack(stage, dim=1))
        controlled = torch.stack(projected, dim=1)
    logits, _ = model._encode_logits_from_plans(
        model._last_slots_wsi,
        model._last_slots_omic,
        _stacked_to_nested(controlled),
    )
    return model._risk(logits).detach(), controlled.detach()


def _formal_evidence_root(output_dir: Path) -> Path | None:
    """Return the nearest evidence root so nested proof artifacts are re-hashed."""

    for candidate in (output_dir, *output_dir.parents):
        if (candidate / "run_manifest.json").is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Top-level audit command
# ---------------------------------------------------------------------------


def cmd_audit(args: argparse.Namespace) -> int:
    parsed = _load_parsed_args(args)
    parsed.k_start = args.fold
    parsed.k_end = args.fold + 1
    parsed.cur_fold = args.fold
    parsed.cur_epoch = int(getattr(args, "epoch", 0))
    parsed.num_workers = 0
    if getattr(args, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model, val_loader, val_data, factory = _load_model_and_loader(args, parsed, args.fold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids: list[str] = []
    factual_risks: list[float] = []
    low_risks: list[float] = []
    high_risks: list[float] = []
    event_times: list[float] = []
    censorships: list[float] = []
    factual_marginal_errors: list[float] = []
    low_marginal_errors: list[float] = []
    high_marginal_errors: list[float] = []
    factual_distance_to_low: list[float] = []
    factual_distance_to_high: list[float] = []
    factual_plan_batches: list[np.ndarray] = []
    low_plan_batches: list[np.ndarray] = []
    high_plan_batches: list[np.ndarray] = []
    controlled_risks: list[float] = []
    controlled_plan_batches: list[np.ndarray] = []

    device = next(model.parameters()).device
    label_df_indexed = val_data.label_df.reset_index(drop=True)
    case_offset = 0
    for batch_idx, data in enumerate(val_loader):
        out, _, event_time, c = _process_data_and_forward(
            parsed, model, data, device, test=False
        )
        logits, _ = out
        explanations = model.last_explanations
        if explanations is None:
            continue
        start = case_offset
        stop = start + len(event_time)
        case_ids.extend(
            label_df_indexed.iloc[start:stop]["case id"].astype(str).tolist()
        )
        factual_risks.extend(_to_numpy(explanations["factual_risk"]).tolist())
        low_risks.extend(_to_numpy(explanations["low_risk_counterfactual"]).tolist())
        high_risks.extend(_to_numpy(explanations["high_risk_counterfactual"]).tolist())
        event_times.extend(_to_numpy(event_time).tolist())
        censorships.extend(_to_numpy(c).tolist())
        factual_marginal_errors.extend(
            _to_numpy(explanations["factual_coupling_marginal_error"]).tolist()
        )
        low_marginal_errors.extend(
            _to_numpy(explanations["low_coupling_marginal_error"]).tolist()
        )
        high_marginal_errors.extend(
            _to_numpy(explanations["high_coupling_marginal_error"]).tolist()
        )
        factual_distance_to_low.extend(
            _to_numpy(explanations["counterfactual_transport_distance_low"]).tolist()
        )
        factual_distance_to_high.extend(
            _to_numpy(explanations["counterfactual_transport_distance_high"]).tolist()
        )
        factual_plan_batches.append(_to_numpy(explanations["factual_transport_plans"]))
        low_plan_batches.append(_to_numpy(explanations["low_transport_plans"]))
        high_plan_batches.append(_to_numpy(explanations["high_transport_plans"]))
        if args.plan_control != "none":
            controlled_risk, controlled_plans = controlled_plan_risk(
                model, args.plan_control, seed=args.control_seed + batch_idx
            )
            controlled_risks.extend(_to_numpy(controlled_risk).tolist())
            controlled_plan_batches.append(_to_numpy(controlled_plans))
        case_offset = stop

    factual_risks_arr = np.asarray(factual_risks, dtype=np.float64)
    low_risks_arr = np.asarray(low_risks, dtype=np.float64)
    high_risks_arr = np.asarray(high_risks, dtype=np.float64)
    event_times_arr = np.asarray(event_times, dtype=np.float64)
    censorships_arr = np.asarray(censorships, dtype=np.float64)

    direction = direction_consistency(
        factual_risks_arr,
        low_risks_arr,
        high_risks_arr,
        event_times_arr,
        censorships_arr,
    )
    empty_plan_shape = (0, 0, 0, 0, 0)
    factual_plans_arr = (
        np.concatenate(factual_plan_batches, axis=0)
        if factual_plan_batches else np.empty(empty_plan_shape, dtype=np.float32)
    )
    low_plans_arr = (
        np.concatenate(low_plan_batches, axis=0)
        if low_plan_batches else np.empty(empty_plan_shape, dtype=np.float32)
    )
    high_plans_arr = (
        np.concatenate(high_plan_batches, axis=0)
        if high_plan_batches else np.empty(empty_plan_shape, dtype=np.float32)
    )
    reconfiguration = plan_total_variation(
        factual_plans_arr, low_plans_arr, high_plans_arr
    )
    controlled_plans_arr = (
        np.concatenate(controlled_plan_batches, axis=0)
        if controlled_plan_batches else np.empty(empty_plan_shape, dtype=np.float32)
    )
    control_summary = None
    if args.plan_control != "none":
        controlled_risks_arr = np.asarray(controlled_risks, dtype=np.float64)
        control_tv = 0.5 * np.abs(controlled_plans_arr - factual_plans_arr).sum(axis=(-1, -2))
        control_summary = {
            "kind": args.plan_control,
            "mean_plan_tv": float(control_tv.mean()),
            "mean_absolute_risk_change": float(
                np.abs(controlled_risks_arr - factual_risks_arr).mean()
            ),
            "per_case_risk_change": (controlled_risks_arr - factual_risks_arr).tolist(),
        }

    record = {
        "fold": int(args.fold),
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "survot_method": parsed.survot_method,
        "case_ids": case_ids,
        "factual_risk": factual_risks_arr,
        "low_risk": low_risks_arr,
        "high_risk": high_risks_arr,
        "event_time": event_times_arr,
        "censorship": censorships_arr,
        "factual_marginal_error": np.asarray(factual_marginal_errors, dtype=np.float64),
        "low_marginal_error": np.asarray(low_marginal_errors, dtype=np.float64),
        "high_marginal_error": np.asarray(high_marginal_errors, dtype=np.float64),
        "factual_transport_plans": factual_plans_arr,
        "low_transport_plans": low_plans_arr,
        "high_transport_plans": high_plans_arr,
        "plan_control": args.plan_control,
        "controlled_risk": np.asarray(controlled_risks, dtype=np.float64),
        "controlled_transport_plans": controlled_plans_arr,
        "factual_distance_to_low_anchor": np.asarray(factual_distance_to_low, dtype=np.float64),
        "factual_distance_to_high_anchor": np.asarray(factual_distance_to_high, dtype=np.float64),
        "direction": direction,
        "reconfiguration": reconfiguration,
        "factual_plan_control": control_summary,
    }
    import pickle
    with open(output_dir / "audit_cases.pkl", "wb") as handle:
        pickle.dump(record, handle)

    metrics = {
        "fold": int(args.fold),
        "checkpoint": str(args.checkpoint),
        "survot_method": parsed.survot_method,
        "anchor_mode": args.anchor_mode,
        "direction_consistency": direction,
        "reconfiguration": reconfiguration,
        "factual_plan_control": control_summary,
        "n_cases": len(case_ids),
        "mean_factual_risk": float(factual_risks_arr.mean()) if factual_risks_arr.size else None,
        "mean_low_risk": float(low_risks_arr.mean()) if low_risks_arr.size else None,
        "mean_high_risk": float(high_risks_arr.mean()) if high_risks_arr.size else None,
        "mean_factual_distance_to_low_anchor": float(np.mean(factual_distance_to_low))
        if factual_distance_to_low
        else None,
        "mean_factual_distance_to_high_anchor": float(np.mean(factual_distance_to_high))
        if factual_distance_to_high
        else None,
        "max_factual_marginal_error": float(np.max(factual_marginal_errors))
        if factual_marginal_errors else None,
        "max_low_marginal_error": float(np.max(low_marginal_errors))
        if low_marginal_errors else None,
        "max_high_marginal_error": float(np.max(high_marginal_errors))
        if high_marginal_errors else None,
    }
    with open(output_dir / "audit_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=float)
    evidence_root = _formal_evidence_root(output_dir)
    if evidence_root is not None:
        audit_key = f"mechanism_audit/{output_dir.name}"
        finalize_fold_evidence(evidence_root, metrics={audit_key: metrics})
    print(json.dumps(metrics, indent=2, default=float))
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Run the alpha-sweep subcommand for dose_monotonicity audit."""
    parsed = _load_parsed_args(args)
    parsed.k_start = args.fold
    parsed.k_end = args.fold + 1
    parsed.cur_fold = args.fold
    parsed.cur_epoch = int(getattr(args, "epoch", 0))
    parsed.num_workers = 0
    if getattr(args, "gpu", None) is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    model, val_loader, val_data, factory = _load_model_and_loader(args, parsed, args.fold)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = tuple(float(value) for value in args.alphas.split(","))
    sweep = _run_alpha_sweep(model, parsed, val_loader, alphas=alphas)
    high_dose = dose_monotonicity(
        sweep["sweep_alphas"], sweep["high_sweep_risks"], increasing=True
    )
    low_dose = dose_monotonicity(
        sweep["sweep_alphas"], sweep["low_sweep_risks"], increasing=False
    )

    import pickle
    with open(output_dir / "dose_sweep.pkl", "wb") as handle:
        pickle.dump(
            {
                "alphas": sweep["sweep_alphas"],
                "high_risks": sweep["high_sweep_risks"],
                "low_risks": sweep["low_sweep_risks"],
                "high_dose": high_dose,
                "low_dose": low_dose,
                "fold": int(args.fold),
                "anchor_mode": args.anchor_mode,
            },
            handle,
        )
    metrics = {
        "fold": int(args.fold),
        "checkpoint": str(args.checkpoint),
        "anchor_mode": args.anchor_mode,
        "high_dose_monotonicity": high_dose,
        "low_dose_monotonicity": low_dose,
    }
    with open(output_dir / "dose_metrics.json", "w") as handle:
        json.dump(metrics, handle, indent=2, default=float)
    evidence_root = _formal_evidence_root(output_dir)
    if evidence_root is not None:
        dose_key = f"dose_audit/{output_dir.name}"
        finalize_fold_evidence(evidence_root, metrics={dose_key: metrics})
    print(json.dumps(metrics, indent=2, default=float))
    return 0


def _reconfiguration_from_distances(
    factual_distance_low: list[float],
    factual_distance_high: list[float],
    margin: float,
) -> dict[str, float]:
    """Proxy for reconfiguration_magnitude using the cached cost-distance stats.

    The full TV distance requires the intervened plans, which are not exposed
    on ``last_explanations``. The cost-distance mean abs diff between factual
    and intervened tensors is already exposed; if it exceeds the v3.8
    reconfiguration margin the intervention moved geometry.
    """
    if not factual_distance_low or not factual_distance_high:
        return {"mean_cost_distance_low": float("nan"),
                "mean_cost_distance_high": float("nan"),
                "above_margin_rate": float("nan"),
                "margin": float(margin)}
    dist_low = np.asarray(factual_distance_low, dtype=np.float64)
    dist_high = np.asarray(factual_distance_high, dtype=np.float64)
    return {
        "mean_cost_distance_low": float(dist_low.mean()),
        "mean_cost_distance_high": float(dist_high.mean()),
        "above_margin_rate": float(((dist_low + dist_high) * 0.5 > margin).mean()),
        "margin": float(margin),
    }


def _load_parsed_args(args: argparse.Namespace):
    config = apply_overrides(load_config(args.config), args.set or [])
    parsed = process_args_extended(config_to_argv(config))
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="audit-dct-reg")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit", help="Run audit on one checkpoint/fold")
    audit.add_argument("--config", required=True)
    audit.add_argument("--checkpoint", required=True)
    audit.add_argument("--fold", type=int, required=True)
    audit.add_argument("--epoch", type=int, default=0)
    audit.add_argument(
        "--output-dir", required=True,
        help="Per-fold evidence directory; standard audit artifact names are written here.",
    )
    audit.add_argument("--gpu", type=int, default=None)
    audit.add_argument(
        "--plan-control",
        choices=("none", "uniform", "shuffled"),
        default="none",
        help="Replace the factual plan with a feasible control before risk decoding.",
    )
    audit.add_argument("--control-seed", type=int, default=1)
    audit.add_argument(
        "--anchor-mode",
        choices=("normal", "swapped"),
        default="normal",
        help="Audit-only high/low anchor identity control.",
    )
    audit.add_argument("--set", action="append", default=[])
    audit.set_defaults(func=cmd_audit)
    sweep = sub.add_parser("sweep", help="Run alpha sweep for dose monotonicity")
    sweep.add_argument("--config", required=True)
    sweep.add_argument("--checkpoint", required=True)
    sweep.add_argument("--fold", type=int, required=True)
    sweep.add_argument("--epoch", type=int, default=0)
    sweep.add_argument("--alphas", default="0.0,0.25,0.5,0.75,1.0")
    sweep.add_argument("--output-dir", required=True)
    sweep.add_argument("--gpu", type=int, default=None)
    sweep.add_argument(
        "--anchor-mode",
        choices=("normal", "swapped"),
        default="normal",
        help="Audit-only high/low anchor identity control.",
    )
    sweep.add_argument("--set", action="append", default=[])
    sweep.set_defaults(func=cmd_sweep)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    os.chdir(REPO_ROOT)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
