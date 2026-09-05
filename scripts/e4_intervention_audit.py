#!/usr/bin/env python3
"""E4: Continuous Intervention Audit for DCT v3.10 experiments.

This script validates that the learned direction constraint produces monotone
risk responses under continuous alpha interventions from low_risk to high_risk
anchors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Add slotspe runtime to path for dataset imports
from survot_rank.training.paths import ensure_slotspe_in_path
ensure_slotspe_in_path()

from dataset.dataset_survival import SurvivalDatasetFactory
from survot_rank.training.model_factory import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="E4 Continuous Intervention Audit")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--variant", type=str, required=True, 
                       choices=["direction_only", "ipcw_only", "full", "nll_only"])
    parser.add_argument("--cancer", type=str, default="blca")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_root", type=str, default="data/dataset_csv")
    parser.add_argument("--wsi_path", type=str, default="data_feats/blca_new_brca_resnet50_non_norm_features")
    parser.add_argument("--num_alphas", type=int, default=11, help="Number of alpha values to test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="results/e4_audit")
    parser.add_argument("--seed", type=int, default=3)
    return parser.parse_args()


def create_args_namespace(variant: str, cancer: str, fold: int, seed: int = 3):
    """Create args namespace matching training configuration."""
    from argparse import Namespace
    
    # Base configuration matching the training setup
    args = Namespace(
        # Core settings
        survot_method="dct_transport_intervention_consistency",
        cancer=cancer,
        split_key=f"splits_{fold}",
        seed=seed,
        
        # Model architecture
        wsi_projection_dim=256,
        slot_dim=256,
        slot_num_wsi=8,
        slot_num_omics=8,
        slot_iters=3,
        slot_hidden_dim=256,
        
        # Pathway settings
        which_pathways="sig_combine",
        pathway_aggregation="attention_mlp",
        
        # DCT settings
        dct_num_stages=4,
        dct_anchor_momentum=0.9,
        dct_mix_ratio=0.5,
        dct_coordinate_temperature=0.2,
        
        # Sinkhorn
        spt_max_iter=100,
        spt_tolerance=1e-4,
        spt_entropy=0.05,
        
        # FET settings
        fet_num_stages=4,
        fet_loss_reduction="mean",
        
        # Objective weights (variant-specific)
        dct_lambda_ipcw_rank=0.0,
        dct_v38_lambda_direction=0.0,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        
        # IPCW settings
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.5,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=0,
        
        # Direction constraint settings (v3.8)
        dct_v38_direction_margin=0.02,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.05,
        dct_v38_alpha_mid=0.5,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=1,
        dct_v38_ramp_epochs=0,
        dct_v38_dose_every=1,
        
        # ETAR (disabled)
        dct_lambda_etar=0.0,
        dct_etar_margin=0.02,
        dct_etar_uncertainty_weight=0.05,
        dct_etar_temperature=0.5,
        dct_etar_evidence_floor=0.1,
        
        # Evidence settings
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_geometry_reliability_strength=0.0,
        dct_geometry_reliability_temperature=0.25,
        dct_coupling_projection_iters=1000,
        dct_coupling_projection_tol=1e-4,
        
        # Ablation flags (all off by default)
        dct_fixed_coupling=False,
        dct_random_anchors=False,
        dct_perm_labels_seed=0,
        dct_stage_jitter_fraction=0.0,
        dct_freeze_source_prototype="",
        
        # Training (not used in inference)
        cur_epoch=0,
    )
    
    # Set variant-specific weights
    if variant == "direction_only":
        args.dct_v38_lambda_direction = 0.05
    elif variant == "ipcw_only":
        args.dct_lambda_ipcw_rank = 0.10
    elif variant == "full":
        args.survot_method = "dct_v310_directional_regularized_transport"
        args.dct_lambda_ipcw_rank = 0.10
        args.dct_v38_lambda_direction = 0.05
    elif variant == "nll_only":
        pass  # All weights are 0
    
    return args


def load_model_and_data(checkpoint_path: str, variant: str, cancer: str, fold: int, 
                       split: str, data_root: str, wsi_path: str, batch_size: int, seed: int):
    """Load model from checkpoint and prepare dataset."""
    
    # Create args namespace
    args = create_args_namespace(variant, cancer, fold, seed)
    
    # Load dataset to get dimensions
    print(f"Loading dataset: {cancer}, fold={fold}, split={split}")
    dataset_factory = SurvivalDatasetFactory(
        study=cancer,
        data_path=data_root,
        rna_format=args.rna_format,
        label_col=args.label_col,
        signature=args.gene_signature,
        which_splits=args.which_splits,
    )
    
    dataset = SurvivalDataset(
        dataset_factory=dataset_factory,
        wsi_path=wsi_path,
        split_key=split,
        fold=fold,
    )
    
    omic_input_dim = dataset.omics_tensor.size(-1) if dataset.omics_tensor is not None else 0
    omic_names = dataset.omic_names
    pathway_names = dataset.pathway_names
    
    # Create model
    print(f"Creating model: {args.survot_method}")
    model = get_model(
        args.survot_method,
        args,
        omic_input_dim=omic_input_dim,
        omic_names=omic_names,
        pathway_names=pathway_names,
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    return model, dataloader, dataset, args


@torch.no_grad()
def compute_factual_embeddings(model, dataloader, device):
    """Extract factual slots, costs, and risks for all patients."""
    model = model.to(device)
    
    all_slots_wsi = []
    all_slots_omic = []
    all_factual_costs = []
    all_factual_risks = []
    all_case_ids = []
    all_event_times = []
    all_censorship = []
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing factual embeddings")):
        # Move batch to device
        x_wsi = batch["x_wsi"].to(device)
        x_omics = {k: v.to(device) if torch.is_tensor(v) else v 
                  for k, v in batch.items() if k.startswith("x_omic")}
        
        # Prepare full batch
        batch_dict = {
            "x_wsi": x_wsi,
            **x_omics,
        }
        if "event_time" in batch:
            batch_dict["event_time"] = batch["event_time"].to(device)
        if "c" in batch:
            batch_dict["c"] = batch["c"].to(device)
        
        # Extract embeddings (similar to forward pass)
        x_wsi_proj = model.wsi_mlp(x_wsi)
        x_omics_encoded = model._encode_omics(batch_dict)
        
        slots_wsi, slots_omic, _, _ = model._encode_transport_slots(
            x_wsi_proj, x_omics_encoded, batch_dict
        )
        
        # Compute factual costs
        factual_costs, rows, cols, evidence_gate = model._cost_tensor(slots_wsi, slots_omic)
        
        # Compute factual plans and risk
        factual_plans, _ = model._plans_from_cost_tensor(
            factual_costs, rows, cols, epoch=999, replay_fixed=False
        )
        factual_logits, _ = model._encode_logits_from_plans(slots_wsi, slots_omic, factual_plans)
        factual_risk = model._risk(factual_logits)
        
        # Store results
        all_slots_wsi.append(slots_wsi.cpu())
        all_slots_omic.append(slots_omic.cpu())
        all_factual_costs.append(factual_costs.cpu())
        all_factual_risks.append(factual_risk.cpu())
        all_case_ids.extend(batch["case_id"])
        
        if "event_time" in batch:
            all_event_times.append(batch["event_time"].cpu())
        if "c" in batch:
            all_censorship.append(batch["c"].cpu())
    
    # Concatenate all batches
    result = {
        "slots_wsi": torch.cat(all_slots_wsi, dim=0),
        "slots_omic": torch.cat(all_slots_omic, dim=0),
        "factual_costs": torch.cat(all_factual_costs, dim=0),
        "factual_risks": torch.cat(all_factual_risks, dim=0),
        "case_ids": all_case_ids,
    }
    
    if all_event_times:
        result["event_times"] = torch.cat(all_event_times, dim=0)
    if all_censorship:
        result["censorship"] = torch.cat(all_censorship, dim=0)
    
    return result


@torch.no_grad()
def run_intervention_sweep(model, embeddings, alphas, device, batch_size=32):
    """Run continuous intervention sweep with alpha interpolation."""
    model = model.to(device)
    
    num_patients = embeddings["factual_costs"].size(0)
    num_alphas = len(alphas)
    num_stages = embeddings["factual_costs"].size(1)
    
    # Get anchor costs from model
    anchor_costs = model.risk_anchor_costs  # [num_stages, 2, 3, num_wsi, num_omic]
    
    # Storage for results
    all_intervened_risks = np.zeros((num_patients, num_alphas))
    
    # Process in batches
    for start_idx in tqdm(range(0, num_patients, batch_size), desc="Running interventions"):
        end_idx = min(start_idx + batch_size, num_patients)
        batch_factual_costs = embeddings["factual_costs"][start_idx:end_idx].to(device)
        batch_slots_wsi = embeddings["slots_wsi"][start_idx:end_idx].to(device)
        batch_slots_omic = embeddings["slots_omic"][start_idx:end_idx].to(device)
        
        bsz = batch_factual_costs.size(0)
        
        # Run interventions for all alphas
        for alpha_idx, alpha in enumerate(alphas):
            # Interpolate costs
            alpha_t = torch.tensor(alpha, device=device, dtype=batch_factual_costs.dtype)
            anchors = anchor_costs.to(device=device, dtype=batch_factual_costs.dtype)
            
            # For each stage, interpolate between low_risk (0) and high_risk (1) anchors
            low_anchor = anchors[:, 0]  # [num_stages, 3, num_wsi, num_omic]
            high_anchor = anchors[:, 1]  # [num_stages, 3, num_wsi, num_omic]
            
            target_anchor = (1 - alpha_t) * low_anchor + alpha_t * high_anchor
            target_anchor = target_anchor.unsqueeze(0).expand(bsz, -1, -1, -1, -1)
            
            # Interpolate factual costs towards target anchor
            intervened_costs = (1 - alpha_t) * batch_factual_costs + alpha_t * target_anchor
            
            # Extract cost components
            rows = intervened_costs[:, :, 0]
            cols = intervened_costs[:, :, 1]
            
            # Re-solve transport with intervened costs
            intervened_plans, _ = model._plans_from_cost_tensor(
                intervened_costs, rows, cols, epoch=999, replay_fixed=False
            )
            
            # Compute intervened risk
            intervened_logits, _ = model._encode_logits_from_plans(
                batch_slots_wsi, batch_slots_omic, intervened_plans
            )
            intervened_risk = model._risk(intervened_logits)
            
            # Store results
            all_intervened_risks[start_idx:end_idx, alpha_idx] = intervened_risk.cpu().numpy()
    
    return all_intervened_risks


def compute_monotonicity_metrics(factual_risks, intervened_risks, alphas):
    """Compute monotonicity violation metrics."""
    num_patients, num_alphas = intervened_risks.shape
    
    # Compute delta risks (change from factual)
    delta_risks = intervened_risks - factual_risks[:, None]
    
    # Check monotonicity: risk should increase with alpha
    violations = 0
    total_pairs = 0
    
    for i in range(num_alphas - 1):
        for j in range(i + 1, num_alphas):
            # Risk at alpha_j should be >= risk at alpha_i
            diff = intervened_risks[:, j] - intervened_risks[:, i]
            violations += (diff < -1e-6).sum()
            total_pairs += num_patients
    
    violation_rate = violations / total_pairs if total_pairs > 0 else 0.0
    
    # Compute correlation between alpha and risk
    alpha_array = np.array(alphas)
    correlations = []
    for p in range(num_patients):
        corr = np.corrcoef(alpha_array, intervened_risks[p])[0, 1]
        if not np.isnan(corr):
            correlations.append(corr)
    
    mean_correlation = np.mean(correlations) if correlations else 0.0
    
    # Compute mean absolute slope
    slopes = []
    for p in range(num_patients):
        slope = (intervened_risks[p, -1] - intervened_risks[p, 0]) / (alphas[-1] - alphas[0])
        slopes.append(slope)
    
    mean_slope = np.mean(slopes)
    
    metrics = {
        "violation_rate": float(violation_rate),
        "mean_correlation": float(mean_correlation),
        "mean_slope": float(mean_slope),
        "num_violations": int(violations),
        "total_pairs": int(total_pairs),
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.variant / args.cancer / f"fold_{args.fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"E4 Continuous Intervention Audit")
    print(f"{'='*80}")
    print(f"Variant: {args.variant}")
    print(f"Cancer: {args.cancer}")
    print(f"Fold: {args.fold}")
    print(f"Split: {args.split}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load model and data
    model, dataloader, dataset, model_args = load_model_and_data(
        args.checkpoint, args.variant, args.cancer, args.fold,
        args.split, args.data_root, args.wsi_path, args.batch_size, args.seed
    )
    
    # Configure train reference (needed for anchor costs)
    if hasattr(dataset, 'train_dataset') and dataset.train_dataset is not None:
        train_event_times = dataset.train_dataset.event_time
        train_censorship = dataset.train_dataset.censorship
        model.configure_train_reference(
            torch.from_numpy(train_event_times),
            torch.from_numpy(train_censorship),
        )
    
    # Extract factual embeddings
    print("\n[1/3] Computing factual embeddings...")
    embeddings = compute_factual_embeddings(model, dataloader, args.device)
    print(f"  Extracted embeddings for {len(embeddings['case_ids'])} patients")
    
    # Run intervention sweep
    print(f"\n[2/3] Running intervention sweep with {args.num_alphas} alpha values...")
    alphas = np.linspace(0.0, 1.0, args.num_alphas)
    intervened_risks = run_intervention_sweep(
        model, embeddings, alphas, args.device, batch_size=args.batch_size
    )
    
    # Compute metrics
    print("\n[3/3] Computing monotonicity metrics...")
    factual_risks = embeddings["factual_risks"].numpy()
    metrics = compute_monotonicity_metrics(factual_risks, intervened_risks, alphas)
    
    print(f"\n{'='*80}")
    print("Results:")
    print(f"{'='*80}")
    print(f"Violation Rate: {metrics['violation_rate']:.4f}")
    print(f"Mean Correlation (alpha vs risk): {metrics['mean_correlation']:.4f}")
    print(f"Mean Slope: {metrics['mean_slope']:.6f}")
    print(f"Violations: {metrics['num_violations']} / {metrics['total_pairs']}")
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        "args": vars(args),
        "metrics": metrics,
        "alphas": alphas.tolist(),
        "case_ids": embeddings["case_ids"],
        "factual_risks": factual_risks.tolist(),
        "intervened_risks": intervened_risks.tolist(),
    }
    
    if "event_times" in embeddings:
        results["event_times"] = embeddings["event_times"].numpy().tolist()
    if "censorship" in embeddings:
        results["censorship"] = embeddings["censorship"].numpy().tolist()
    
    output_file = output_dir / f"e4_results_{args.split}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    return metrics


if __name__ == "__main__":
    main()
