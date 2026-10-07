#!/usr/bin/env python3
"""
Simple reader ablation test - bypass the complex argument parsing.
Usage: python scripts/test_reader_ablation_simple.py <checkpoint_path> <mode>
Where mode is: full, pair_context_only, or plan_only
"""
import sys
import os
from pathlib import Path

# Set up paths before any imports
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Now we can import after setting up environment
from scripts.audit_dct_reg import _to_numpy
import torch
import torch.nn.functional as F
import numpy as np

def hijack_fusion(model, mode):
    """Modify fusion module for ablation."""
    original_forward = model.fusion.forward
    
    if mode == "full":
        # No change - use original
        return
    
    elif mode == "pair_context_only":
        def ablated_forward(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
            # Build pair_context but DON'T use plans
            bsz, sw, dim = slots_wsi.shape
            so = slots_omic.size(1)
            
            # Build pair tokens (this includes pair_context)
            pair_tokens = model.fusion._build_pair_tokens(slots_wsi, slots_omic)
            pair_context = pair_tokens[..., : dim * 3]
            
            # Use ZERO cost encoding (bypass all plans)
            cost_encoding = torch.zeros(bsz, sw, so, dim, device=slots_wsi.device)
            
            # Concatenate and project
            cost_concat = torch.cat([cost_encoding, pair_context], dim=-1)
            pair_tokens = model.fusion.proj(cost_concat)
            
            # Rest is standard
            pair_tokens = pair_tokens.reshape(bsz, sw * so, dim)
            # Use uniform mass instead of plan
            pair_mass = torch.zeros(bsz, sw * so, 1, device=slots_wsi.device)
            q = F.normalize(model.fusion.event_queries, dim=-1)
            t = F.normalize(pair_tokens, dim=-1)
            scores = torch.einsum("kd,bpd->bpk", q, t)
            scores = scores + pair_mass
            assign = torch.softmax(scores.transpose(1, 2), dim=-1)
            events = torch.bmm(assign, pair_tokens)
            events = model.fusion.norm(model.fusion.cross_attn(events))
            return events + model.fusion.refine(events), assign
        
        model.fusion.forward = ablated_forward
    
    elif mode == "plan_only":
        def ablated_forward(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
            # Use plans but NO pair_context
            bsz, sw, dim = slots_wsi.shape
            so = slots_omic.size(1)
            
            # Encode costs from plans (standard)
            c_cos = model.fusion.cost_convs["cosine"](plan_cos.unsqueeze(-1))
            c_euc = model.fusion.cost_convs["euclidean"](plan_euc.unsqueeze(-1))
            c_dot = model.fusion.cost_convs["dot"](plan_dot.unsqueeze(-1))
            cost_encoding = c_cos + c_euc + c_dot
            
            # Use ZERO pair_context (no direct connection)
            pair_context = torch.zeros(bsz, sw, so, dim * 3, device=slots_wsi.device)
            
            # Concatenate and project
            cost_concat = torch.cat([cost_encoding, pair_context], dim=-1)
            pair_tokens = model.fusion.proj(cost_concat)
            
            # Rest is standard
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


def compute_cindex(risks, times, censorships):
    """Compute C-index."""
    risks = _to_numpy(risks).flatten()
    times = _to_numpy(times).flatten()
    censorships = _to_numpy(censorships).flatten()
    
    n_pairs = 0
    n_concordant = 0
    
    for i in range(len(risks)):
        if censorships[i] > 0.5:
            continue
        for j in range(len(risks)):
            if times[i] < times[j]:
                n_pairs += 1
                if risks[i] > risks[j]:
                    n_concordant += 1
    
    return n_concordant / n_pairs if n_pairs > 0 else 0.5


def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/test_reader_ablation_simple.py <checkpoint_path> <mode>")
        print("Modes: full, pair_context_only, plan_only")
        return 1
    
    checkpoint_path = sys.argv[1]
    mode = sys.argv[2]
    fold = 1
    
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"Mode: {mode}")
    
    # Find config file in the evidence directory
    ckpt_path = Path(checkpoint_path)
    evidence_dir = ckpt_path.parent
    config_path = evidence_dir / "resolved_config.yaml"
    
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return 1
    
    print(f"Using config: {config_path}")
    
    # Load config and create parsed args
    from survot_rank.config import load_config, config_to_argv, apply_overrides
    from survot_rank.training.extended_args import process_args_extended
    
    # CRITICAL: Clear sys.argv before calling process_args_extended
    # Otherwise it will try to parse our custom arguments
    original_argv = sys.argv
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    config = load_config(config_path)
    parsed = process_args_extended(config_to_argv(config))
    
    # Restore sys.argv
    sys.argv = original_argv
    
    # Create fake args for the loader
    class FakeArgs:
        pass
    
    args = FakeArgs()
    args.checkpoint = checkpoint_path
    
    # Load model using audit script's function
    from scripts.audit_dct_reg import _load_model_and_loader
    
    model, val_loader, val_data, factory = _load_model_and_loader(args, parsed, fold)
    device = next(model.parameters()).device
    
    print(f"Loaded! Device: {device}")
    print(f"Val batches: {len(val_loader)}")
    
    # Hijack fusion
    hijack_fusion(model, mode)
    print(f"Fusion hijacked for mode: {mode}")
    
    # Evaluate
    all_risks = []
    all_times = []
    all_censorships = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(val_loader):
            x_wsi = data["x_path"][0].to(device)
            x_omic = data["x_omic"].to(device)
            event_time = data["survival"].to(device)
            c = data["censorship"].to(device)
            
            hazards, _, _, _ = model(x_wsi=x_wsi, x_omic=x_omic)
            risk = -hazards.sum(dim=1)
            
            all_risks.append(risk.cpu())
            all_times.append(event_time.cpu())
            all_censorships.append(c.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(val_loader)} batches")
    
    all_risks = torch.cat(all_risks)
    all_times = torch.cat(all_times)
    all_censorships = torch.cat(all_censorships)
    
    cindex = compute_cindex(all_risks, all_times, all_censorships)
    
    print(f"\n{'='*60}")
    print(f"Mode: {mode}")
    print(f"C-index: {cindex:.4f}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
