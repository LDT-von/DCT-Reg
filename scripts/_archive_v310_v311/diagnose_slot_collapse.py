#!/usr/bin/env python3
"""诊断脚本：检查 slot_attention 输出的坍缩源"""
import torch
import sys
sys.path.insert(0, "/data1/DCT-Reg")
from survot_rank.research.methods.legacy.experimental.dct_v311_slot_interpretable.model import DCTV311SlotInterpretable
from types import SimpleNamespace

# 创建一个简单 args 复现
def make_args():
    args = SimpleNamespace(
        study="blca",
        data_root_dir="/data/CPathPatchFeature",
        data_path="/data1/dataset_csv",
        batch_size=4,
        num_classes=4,
        wsi_projection_dim=256,
        slot_num_wsi=8,
        slot_num_omics=8,
        wsi_encoder="uni",
        encoding_dim=1024,
        # 必需参数
        ot_eps=0.05,
        ot_max_iter=50,
        ot_heads=4,
        ot_layers=2,
        ot_dropout=0.15,
        dct_num_stages=4,
        dct_lambda_ipcw_rank=0.10,
        dct_anchor_momentum=0.90,
        spt_prog_cost=0.20,
        rg_eps_start=0.10,
        rg_eps_anneal=12,
        dct_v311_lambda_slot_nll=0.05,
        dct_v311_lambda_slot_diversity=0.10,
        dct_v311_variance_min=0.001,
        dct_v311_variance_max=0.050,
        dct_v38_lambda_direction=0.0,
        dct_v38_lambda_dose=0.0,
        dct_v38_lambda_reconfiguration=0.0,
        dct_evidence_cost_weight=0.0,
        dct_evidence_mass_floor=0.05,
        dct_evidence_marginal_strength=1.0,
        dct_geometry_reliability_strength=0.0,
        dct_coupling_projection_iters=100,
        dct_coupling_projection_tol=1e-4,
        dct_coordinate_temperature=0.30,
        dct_mix_ratio=1.0,
        dct_v38_direction_margin=0.02,
        dct_v38_dose_margin=0.005,
        dct_v38_reconfiguration_margin=0.02,
        dct_v38_temperature=0.05,
        dct_v38_alpha_mid=0.5,
        dct_v38_alpha_full=1.0,
        dct_v38_warmup_epochs=0,
        dct_v38_ramp_epochs=0,
        dct_v38_dose_every=1,
        dct_lambda_etar=0.0,
        dct_lambda_listwise=0.0,
        dct_ipcw_rank_margin=0.02,
        dct_ipcw_rank_temperature=0.5,
        dct_ipcw_max_weight=10.0,
        dct_ipcw_rank_memory_size=64,
        dct_slot_init_mode="deterministic",
        n_classes=4,
        # FROZEN
    )
    # apply frozen arguments
    from survot_rank.research.methods.dct_v310_directional_regularized_transport.model import DCTV310DirectionalRegularizedTransport
    frozen = DCTV311SlotInterpretable.FROZEN_ARGUMENTS
    for k, v in frozen.items():
        setattr(args, k, v)
    return args

args = make_args()

# 加载 model
print("Building model...")
model = DCTV311SlotInterpretable(args, omic_input_dim=320, omic_names=[], pathway_names=[])
model.eval()

# Dummy data
B = 4
x_wsi = torch.randn(B, 100, 1024)  # [B, N, D]
x_omic = torch.randn(B, 320)
mask = torch.ones(B, 100, dtype=torch.bool)

# 手动 forward
with torch.no_grad():
    x_wsi_proj = model.wsi_mlp(x_wsi)
    print(f"x_wsi_proj shape: {x_wsi_proj.shape}")
    print(f"x_wsi_proj std: {x_wsi_proj.std().item():.4f}")

    local_slots_wsi = model.slot_attention_wsi(x_wsi_proj)
    print(f"\nlocal_slots_wsi shape: {local_slots_wsi.shape}")
    print(f"  std across slots (dim=1): {local_slots_wsi.std(dim=1).mean().item():.4f}")
    print(f"  std across features (dim=2): {local_slots_wsi.std(dim=2).mean().item():.4f}")

    # semantic slots after prototype projection
    slots_wsi, _ = model._semantic_slots(local_slots_wsi, model.shared_wsi_prototypes)
    print(f"\nslots_wsi (after semantic) shape: {slots_wsi.shape}")
    print(f"  std across slots (dim=1): {slots_wsi.std(dim=1).mean().item():.4f}")
    print(f"  std across features (dim=2): {slots_wsi.std(dim=2).mean().item():.4f}")

    # Now the hazard
    hazard_wsi = torch.sigmoid(model.per_slot_hazard_wsi(slots_wsi))
    print(f"\nhazard_wsi shape: {hazard_wsi.shape}")
    print(f"  range: [{hazard_wsi.min().item():.4f}, {hazard_wsi.max().item():.4f}]")
    print(f"  mean: {hazard_wsi.mean().item():.4f}")
    var_per_sample = []
    for b in range(B):
        v = hazard_wsi[b].var()
        var_per_sample.append(v.item())
    print(f"  per-sample variance: {[f'{v:.6f}' for v in var_per_sample]}")
    print(f"  mean variance: {sum(var_per_sample)/len(var_per_sample):.6f}")
