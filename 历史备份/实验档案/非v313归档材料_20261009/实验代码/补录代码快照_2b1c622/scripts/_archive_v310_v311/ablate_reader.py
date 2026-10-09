#!/usr/bin/env python3
"""Reader ablation: test full / pair-context-only / plan-only modes.

This script tests whether the risk reader primarily relies on pair-context
bypass (slots_wsi ⊗ slots_omic) or the OT transport plans.

Modes:
  - full: Normal reader with both pair-context and plan encoding
  - pair_context_only: Remove plan input, only use slot pairs
  - plan_only: Remove pair-context bypass, only use plan encoding

Expected finding:
  If pair_context_only ≈ full, it proves the OT plan is being bypassed.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survot_rank.cli import add_project_paths
add_project_paths()

from survot_rank.config import apply_overrides, config_to_argv, load_config
from survot_rank.training.extended_args import process_args_extended
from survot_rank.training.model_factory import get_model

try:
    from survot_rank.research.legacy.slotspe_runtime.dataset.dataset_survival import (
        SurvivalDatasetFactory,
    )
except ImportError:
    SurvivalDatasetFactory = None

try:
    from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (
        _process_data_and_forward,
    )
except ImportError:
    _process_data_and_forward = None

try:
    from survot_rank.training.sparse_event import get_split
except ImportError:
    get_split = None
if get_split is None:
    from survot_rank.training.train_runner import get_split


def _load_checkpoint_and_data(checkpoint_path, cancer, fold):
    """Load model and data - adapted from audit_dct_reg.py."""
    ckpt_path = Path(checkpoint_path)
    evidence_dir = ckpt_path.parent
    
    # Try to load config from checkpoint directory or use default
    config_path = evidence_dir / "resolved_config.yaml"
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "dct_v310_directional_regularized_transport.yaml"
    
    config = load_config(config_path)
    
    # Save and temporarily remove our custom args from sys.argv
    import sys
    original_argv = sys.argv.copy()
    # Remove our custom arguments before calling process_args_extended
    filtered_argv = [original_argv[0]]  # Keep script name
    i = 1
    while i < len(original_argv):
        arg = original_argv[i]
        if arg in ("--checkpoint", "--cancer", "--fold", "--modes", "--output", "--gpu"):
            i += 2  # Skip this arg and its value
        else:
            filtered_argv.append(arg)
            i += 1
    sys.argv = filtered_argv
    
    argv = config_to_argv(config)
    parsed = process_args_extended(argv)
    
    # Restore original argv
    sys.argv = original_argv
    
    parsed.cur_fold = fold
    
    # Load data factory
    factory = SurvivalDatasetFactory(
        parsed.study,
        parsed.which_splits,
        parsed.data_path,
        parsed.data_root_dir,
        label_col=parsed.label_col,
        ignore_missing_moltype=False,
        ignore_missing_wsi=True,
        on_missing_wsi=parsed.on_missing_wsi,
        feature_format=parsed.wsi_encoder,
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
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, parsed, val_loader, val_data, factory


def _hijack_fusion_forward(model, mode):
    """Hijack fusion forward to test different ablation modes."""
    original_forward = model.fusion.forward
    
    if mode == "full":
        # No hijack needed
        return
    
    elif mode == "pair_context_only":
        # Remove plan encoding, only use pair context
        def ablated_forward(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
            bsz, sw, dim = slots_wsi.shape
            so = slots_omic.shape[1]
            
            # Build pair tokens (the bypass)
            pair_tokens = model.fusion._build_pair_tokens(slots_wsi, slots_omic)
            
            # Project to dim, but WITHOUT plan encoding
            pair_context = pair_tokens[..., : dim * 3]
            pair_tokens_proj = model.fusion.proj(pair_context)  # Skip cost encoding
            
            # Rest is the same
            pair_tokens_proj = pair_tokens_proj.reshape(bsz, sw * so, dim)
            pair_mass = plan_cos.reshape(bsz, sw * so).clamp_min(1e-8).log().unsqueeze(-1)
            q = F.normalize(model.fusion.event_queries, dim=-1)
            t = F.normalize(pair_tokens_proj, dim=-1)
            scores = torch.einsum("kd,bpd->bpk", q, t)
            scores = scores + pair_mass
            assign = torch.softmax(scores.transpose(1, 2), dim=-1)
            events = torch.bmm(assign, pair_tokens_proj)
            events = model.fusion.norm(model.fusion.cross_attn(events))
            return events + model.fusion.refine(events), assign
        
        model.fusion.forward = ablated_forward
    
    elif mode == "plan_only":
        # Remove pair context, only use plan encoding
        def ablated_forward(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
            bsz, sw, dim = slots_wsi.shape
            so = slots_omic.shape[1]
            
            # Project each cost type WITHOUT pair context
            c_cos = model.fusion.cost_convs["cosine"](plan_cos.unsqueeze(-1))
            c_euc = model.fusion.cost_convs["euclidean"](plan_euc.unsqueeze(-1))
            c_dot = model.fusion.cost_convs["dot"](plan_dot.unsqueeze(-1))
            cost_concat = torch.cat([c_cos, c_euc, c_dot], dim=-1)
            pair_tokens = model.fusion.proj(cost_concat)  # No pair_context addition
            
            # Rest is the same
            pair_tokens = pair_tokens.reshape(bsz, sw * so, dim)
            pair_mass = plan_cos.reshape(bsz, sw * so).clamp_min(1e-8).log().unsqueeze(-1)
            q = F.normalize(model.fusion.event_queries, dim=-1)
            t = F.normalize(pair_tokens, dim=-1)
            scores = torch.einsum("kd,bpd->bpk", q, t)
            scores = scores + pair_mass
            assign = torch.softmax(scores.transpose(1, 2), dim=-1)
            events = torch.bmm(assign, pair_tokens)
            events = model.fusion.norm(model.fusion.cross_attn(events))
            return events + model.fusion.refine(events), assign
        
        model.fusion.forward = ablated_forward
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _process_data_and_forward(parsed, model, data, device, test=False):
    """Forward pass through model."""
    x_wsi = data["x_path"][0].to(device)
    x_omic = data["x_omic"].to(device) if parsed.use_omic else None
    event_time = data["survival"].to(device)
    c = data["censorship"].to(device)
    
    with torch.no_grad():
        kwargs = {"x_wsi": x_wsi}
        if x_omic is not None:
            kwargs["x_omic"] = x_omic
        if not test:
            kwargs["label"] = event_time
            kwargs["event_time"] = event_time
            kwargs["c"] = c
        out = model(**kwargs)
    
    return out, x_wsi, x_omic, event_time, c


def _compute_cindex(risks, event_times, censorships):
    """Compute concordance index."""
    n_pairs = 0
    n_concordant = 0
    
    for i in range(len(risks)):
        if censorships[i] > 0.5:  # censored
            continue
        for j in range(len(risks)):
            if event_times[i] < event_times[j]:
                n_pairs += 1
                if risks[i] > risks[j]:
                    n_concordant += 1
    
    return n_concordant / n_pairs if n_pairs > 0 else 0.5


def evaluate_mode(model, parsed, val_loader, mode, device):
    """Evaluate model in a specific ablation mode."""
    print(f"\n{'='*60}")
    print(f"Evaluating mode: {mode.upper()}")
    print(f"{'='*60}")
    
    # Hijack fusion
    _hijack_fusion_forward(model, mode)
    
    all_risks = []
    all_event_times = []
    all_censorships = []
    
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            out, _, _, event_time, c = _process_data_and_forward(
                parsed, model, data, device, test=False
            )
            logits, _ = out
            risk = model._risk(logits)
            
            all_risks.extend(risk.cpu().numpy().tolist())
            all_event_times.extend(event_time.cpu().numpy().tolist())
            all_censorships.extend(c.cpu().numpy().tolist())
    
    cindex = _compute_cindex(all_risks, all_event_times, all_censorships)
    
    print(f"C-index: {cindex:.4f}")
    print(f"Samples: {len(all_risks)}")
    
    return {
        "mode": mode,
        "cindex": float(cindex),
        "n_samples": len(all_risks),
    }


def main():
    # Use a simple parser that won't conflict with process_args_extended
    import sys
    
    checkpoint_path = None
    cancer = "blca"
    fold = 0
    modes_str = "full,pair_context_only,plan_only"
    output = None
    gpu = 0
    
    # Simple command-line parsing
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--checkpoint":
            checkpoint_path = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--cancer":
            cancer = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--fold":
            fold = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == "--modes":
            modes_str = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--output":
            output = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--gpu":
            gpu = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    if checkpoint_path is None:
        print("Error: --checkpoint is required")
        return 1
    
    class Args:
        pass
    
    args = Args()
    args.checkpoint = checkpoint_path
    args.cancer = cancer
    args.fold = fold
    args.modes = modes_str
    args.output = output
    args.gpu = gpu
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and data
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model, parsed, val_loader, val_data, factory = _load_checkpoint_and_data(
        args.checkpoint, args.cancer, args.fold
    )
    model = model.to(device)
    
    print(f"\nLoaded {args.cancer} fold {args.fold}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Run ablations
    modes = [m.strip() for m in args.modes.split(",")]
    results = []
    
    for mode in modes:
        result = evaluate_mode(model, parsed, val_loader, mode, device)
        results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    full_cindex = None
    for r in results:
        print(f"{r['mode']:25s}: C-index = {r['cindex']:.4f}")
        if r['mode'] == 'full':
            full_cindex = r['cindex']
    
    if full_cindex is not None:
        print(f"\n{'='*60}")
        print("RELATIVE TO FULL MODEL")
        print(f"{'='*60}")
        for r in results:
            if r['mode'] != 'full':
                diff = r['cindex'] - full_cindex
                pct = 100 * diff / full_cindex
                print(f"{r['mode']:25s}: {diff:+.4f} ({pct:+.1f}%)")
    
    # Save results
    output = {
        "checkpoint": str(args.checkpoint),
        "cancer": args.cancer,
        "fold": args.fold,
        "results": results,
    }
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
