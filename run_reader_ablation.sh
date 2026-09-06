#!/bin/bash
# Simple wrapper to run reader ablation without argument conflicts

CKPT="$1"
MODE="$2"

if [ -z "$CKPT" ] || [ -z "$MODE" ]; then
    echo "Usage: $0 <checkpoint_path> <mode>"
    echo "Modes: full, pair_context_only, plan_only"
    exit 1
fi

# Find config in evidence directory
EVIDENCE_DIR=$(dirname "$CKPT")
CONFIG="$EVIDENCE_DIR/resolved_config.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config not found at $CONFIG"
    exit 1
fi

echo "Checkpoint: $CKPT"
echo "Config: $CONFIG"
echo "Mode: $MODE"
echo ""

# Create a temporary Python script that won't conflict with argparse
TEMP_SCRIPT=$(mktemp /tmp/ablation_XXXXXX.py)

cat > "$TEMP_SCRIPT" << 'EOPYTHON'
import sys
import os
from pathlib import Path

# Get arguments before any imports that might pollute argv
checkpoint_path = sys.argv[1]
config_path = sys.argv[2]
mode = sys.argv[3]
fold = 1

# Now set up environment - use absolute path
REPO_ROOT = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Clear argv completely before imports
sys.argv = [sys.argv[0]]

import torch
import torch.nn.functional as F
import numpy as np

from survot_rank.cli import add_project_paths
add_project_paths()

from survot_rank.config import load_config, config_to_argv
from survot_rank.training.extended_args import process_args_extended
from scripts.audit_dct_reg import _load_model_and_loader, _to_numpy

def hijack_fusion(model, mode):
    """Modify fusion module for ablation."""
    if mode == "full":
        return
    
    elif mode == "pair_context_only":
        def ablated_forward(slots_wsi, slots_omic, plan_cos, plan_euc, plan_dot):
            bsz, sw, dim = slots_wsi.shape
            so = slots_omic.size(1)
            pair_tokens = model.fusion._build_pair_tokens(slots_wsi, slots_omic)
            pair_context = pair_tokens[..., : dim * 3]
            cost_encoding = torch.zeros(bsz, sw, so, dim, device=slots_wsi.device)
            cost_concat = torch.cat([cost_encoding, pair_context], dim=-1)
            pair_tokens = model.fusion.proj(cost_concat)
            pair_tokens = pair_tokens.reshape(bsz, sw * so, dim)
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
            bsz, sw, dim = slots_wsi.shape
            so = slots_omic.size(1)
            c_cos = model.fusion.cost_convs["cosine"](plan_cos.unsqueeze(-1))
            c_euc = model.fusion.cost_convs["euclidean"](plan_euc.unsqueeze(-1))
            c_dot = model.fusion.cost_convs["dot"](plan_dot.unsqueeze(-1))
            cost_encoding = c_cos + c_euc + c_dot
            pair_context = torch.zeros(bsz, sw, so, dim * 3, device=slots_wsi.device)
            cost_concat = torch.cat([cost_encoding, pair_context], dim=-1)
            pair_tokens = model.fusion.proj(cost_concat)
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

def compute_cindex(risks, times, censorships):
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

# Load config
config = load_config(config_path)
parsed = process_args_extended(config_to_argv(config))

class FakeArgs:
    pass
args = FakeArgs()
args.checkpoint = checkpoint_path

model, val_loader, val_data, factory = _load_model_and_loader(args, parsed, fold)
device = next(model.parameters()).device

print(f"Loaded! Device: {device}, Val batches: {len(val_loader)}")

hijack_fusion(model, mode)
print(f"Fusion hijacked for mode: {mode}\n")

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
EOPYTHON

# Run the temp script
cd /data1/DCT-Reg
python "$TEMP_SCRIPT" "$CKPT" "$CONFIG" "$MODE"
EXIT_CODE=$?

# Clean up
rm -f "$TEMP_SCRIPT"

exit $EXIT_CODE
