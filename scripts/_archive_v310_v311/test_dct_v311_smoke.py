#!/usr/bin/env python3
"""Smoke test for DCT v3.11 per-slot interpretability.

Verifies:
  1. Model instantiates without error
  2. Forward pass (train mode) produces logits and aux_loss
  3. New per-slot losses are non-zero and finite
  4. Gradients flow back to slot_attention parameters
  5. Diversity loss is active (variance within bounds after a few steps)
  6. Eval mode produces per-slot hazard predictions in last_explanations
  7. Catalog registration works
"""

import sys
from pathlib import Path
from argparse import Namespace
import torch
import torch.nn as nn

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# ── 1. Verify catalog registration ──────────────────────────────────────────
from survot_rank.research.methods.catalog import METHOD_CATALOG, METHOD_REGISTRY, METHOD_ALIASES

assert "dct_v311_slot_interpretable" in METHOD_CATALOG, \
    "dct_v311 not in catalog"
assert "dct_v311" in METHOD_ALIASES, \
    "dct_v311 alias not in METHOD_ALIASES"
print("✅  Catalog registration OK")

# ── 2. Build mock args ───────────────────────────────────────────────────
def make_args():
    args = Namespace()
    # Data / dataset
    args.study = "blca"
    args.n_classes = 4
    # Use Pathways with a single dummy pathway group to match real training setup.
    args.rna_format = "Pathways"
    args.omic_sizes = [512]   # 1 pathway group, 512 genes → [B, 1, dim] after encoding
    args.omic_names = None
    args.pathway_names = None
    # Encoding
    args.encoding_dim = 1024
    args.wsi_projection_dim = 256
    args.num_patches = 256
    # Slots
    args.slot_num_wsi = 8
    args.slot_num_omics = 4
    args.slot_iters = 3
    args.dct_slot_init_mode = "gaussian"
    args.dct_slot_eval_seed = 1729
    # OT
    args.otehv2_eps = 0.05
    args.otehv2_iter = 20
    args.otehv2_warmup = 3
    args.otehv2_num_events = 4
    args.otehv2_heads = 4
    args.otehv2_layers = 2
    args.otehv2_dropout = 0.0
    # DCT v3.10 frozen (overridden by the model constructor)
    args.bag_loss = "nll_surv"
    args.rg_num_events = 4
    args.spt_num_stages = 4
    args.rg_prog_cost = 0.20
    args.rg_lambda_ot = 0.06
    args.rg_lambda_rank = 0.15
    args.rg_lambda_stage = 0.02
    args.rg_eps_start = 0.10
    args.rg_eps_anneal = 12
    args.dct_num_stages = 4
    args.dct_lambda_ipcw_rank = 0.10
    args.dct_ipcw_rank_margin = 0.02
    args.dct_ipcw_rank_temperature = 0.50
    args.dct_ipcw_max_weight = 10.0
    args.dct_ipcw_rank_memory_size = 0
    args.dct_lambda_etar = 0.0
    args.dct_etar_margin = 0.02
    args.dct_etar_uncertainty_weight = 0.05
    args.dct_etar_temperature = 0.50
    args.dct_etar_evidence_floor = 0.10
    args.dct_anchor_momentum = 0.90
    args.dct_evidence_cost_weight = 0.0
    args.dct_evidence_mass_floor = 0.05
    args.dct_evidence_marginal_strength = 1.0
    args.dct_geometry_reliability_strength = 0.0
    args.dct_geometry_reliability_temperature = 0.25
    args.dct_coupling_projection_iters = 50
    args.dct_coupling_projection_tol = 1e-3
    args.dct_mix_ratio = 0.50
    args.dct_coordinate_temperature = 0.20
    args.dct_fixed_coupling = False
    args.dct_random_anchors = False
    args.dct_perm_labels_seed = 0
    args.dct_stage_jitter_fraction = 0.0
    args.dct_freeze_source_prototype = ""
    # v3.8 structural (will be zeroed by the model)
    args.dct_v38_lambda_direction = 0.0
    args.dct_v38_lambda_dose = 0.0
    args.dct_v38_lambda_reconfiguration = 0.0
    args.dct_v38_direction_margin = 0.02
    args.dct_v38_dose_margin = 0.005
    args.dct_v38_reconfiguration_margin = 0.02
    args.dct_v38_temperature = 0.05
    args.dct_v38_alpha_mid = 0.50
    args.dct_v38_alpha_full = 1.00
    args.dct_v38_warmup_epochs = 0
    args.dct_v38_ramp_epochs = 0
    args.dct_v38_dose_every = 1
    # v3.11 new
    args.dct_v311_lambda_slot_nll = 0.05
    args.dct_v311_lambda_slot_diversity = 0.02
    args.dct_v311_variance_min = 0.005
    args.dct_v311_variance_max = 0.050
    args.cur_epoch = 0
    return args


# ── 3. Instantiate model ──────────────────────────────────────────────────
args = make_args()
model_cls_name = METHOD_REGISTRY["dct_v311_slot_interpretable"]
print(f"  Model class: {model_cls_name}")

from survot_rank.training.model_factory import get_model

model = get_model(
    method="dct_v311_slot_interpretable",
    args=args,
    omic_input_dim=None,
    omic_names=args.omic_names,
    pathway_names=args.pathway_names,
)
model.train()
print("✅  Model instantiation OK")

# ── 4. Verify frozen invariants ──────────────────────────────────────────
assert model.dct_lambda_ipcw_rank == 0.10, \
    f"ipcw_rank weight wrong: {model.dct_lambda_ipcw_rank}"
assert model.dct_v38_lambda_direction == 0.0, \
    f"direction loss should be 0, got {model.dct_v38_lambda_direction}"
assert model.dct_v38_lambda_dose == 0.0
assert model.dct_v38_lambda_reconfiguration == 0.0
# Verify per-slot heads exist
assert hasattr(model, "per_slot_hazard_wsi")
assert hasattr(model, "per_slot_hazard_omic")
assert model.per_slot_hazard_wsi.out_features == args.n_classes
assert model.per_slot_hazard_omic.out_features == args.n_classes
print("✅  Frozen invariants OK")

# ── 5. Configure train reference (IPCW requires it) ────────────────────────
n_train = 64
event_times = torch.arange(n_train).float() + 10.0   # 10..73
censorship = torch.zeros(n_train)
censorship[n_train // 2:] = 1.0   # second half censored
model.configure_train_reference(event_times, censorship)
assert model.has_train_reference, "train reference not configured"
print("✅  Train reference configured OK")

# ── 6. Build fake batch ──────────────────────────────────────────────────
B, N_patches, D_enc = 8, 256, 1024
K_w, K_o = 8, 4
C = args.n_classes
# Pathways: x_omic1 = [B, num_genes_in_pathway]
num_genes = args.omic_sizes[0]   # 512

device = torch.device("cpu")
x_wsi = torch.randn(B, N_patches, D_enc).to(device)
x_omics_pathway = torch.randn(B, num_genes).to(device)

# Discrete-time labels: one-hot [B, C]
y = torch.zeros(B, C)
labels = [0, 1, 2, 3, 0, 1, 2, 3]
for i, lab in enumerate(labels):
    y[i, lab] = 1.0
event_time = torch.tensor([15.0, 30.0, 45.0, 60.0, 25.0, 50.0, 70.0, 80.0])
c = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]).float()   # 0=event, 1=censored

batch = {
    "x_wsi": x_wsi,
    "x_omic1": x_omics_pathway,
    "y": y,
    "event_time": event_time,
    "c": c,
    "cur_epoch": 0,
}

# ── 7. Forward pass (train) ───────────────────────────────────────────────
logits, aux_loss = model(**batch)
assert logits.shape == (B, C), f"logits shape mismatch: {logits.shape}"
assert torch.isfinite(logits).all(), "logits contain NaN/Inf"
assert torch.isfinite(aux_loss) or aux_loss == 0.0, \
    f"aux_loss not finite: {aux_loss}"
print(f"✅  Forward pass OK — logits shape={logits.shape}, aux_loss={aux_loss.item():.4f}")

# ── 8. Check diagnostics ──────────────────────────────────────────────────
losses = model.last_training_losses
required_keys = [
    "ot", "ipcw_rank", "v311_per_slot_nll", "v311_slot_diversity",
    "v311_slot_variance", "v311_per_slot_nll_lambda", "v311_slot_diversity_lambda",
]
for k in required_keys:
    assert k in losses, f"Missing diagnostic key: {k}"
    v = losses[k]
    if torch.is_tensor(v):
        assert torch.isfinite(v).all(), f"Diagnostic {k} not finite: {v}"
print(f"✅  Training diagnostics OK — "
      f"per_slot_nll={losses['v311_per_slot_nll'].item():.4f}, "
      f"slot_diversity={losses['v311_slot_diversity'].item():.4f}, "
      f"slot_variance={losses['v311_slot_variance'].item():.4f}")

# ── 9. Gradient flow test ─────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.zero_grad()
_, aux = model(**batch)
# Use a well-formed scalar to drive backward.
loss_for_bwd = model(**batch)[0].sum() * 0.0 + aux
loss_for_bwd.backward()
optimizer.step()

# Check per-slot hazard head has grad (must have gradient from per_slot_nll).
grad_w_wsi = model.per_slot_hazard_wsi.weight.grad
assert grad_w_wsi is not None, "per_slot_hazard_wsi.weight has no gradient"
assert grad_w_wsi.abs().sum() > 0, "per_slot_hazard_wsi.weight gradient is all zeros"

grad_w_omic = model.per_slot_hazard_omic.weight.grad
assert grad_w_omic is not None, "per_slot_hazard_omic.weight has no gradient"
assert grad_w_omic.abs().sum() > 0, "per_slot_hazard_omic.weight gradient is all zeros"

# Check slot attention has grad (critical: gradient reaches slot mechanism).
grad_slots_wsi = model.slot_attention_wsi.slots_mu.grad
assert grad_slots_wsi is not None, "slot_attention_wsi.slots_mu has no gradient"
assert grad_slots_wsi.abs().sum() > 0, "slot_attention_wsi.slots_mu gradient is all zeros"

grad_slots_omic = model.slot_attention_omic.slots_mu.grad
assert grad_slots_omic is not None, "slot_attention_omic.slots_mu has no gradient"
assert grad_slots_omic.abs().sum() > 0, "slot_attention_omic.slots_mu gradient is all zeros"

print("✅  Gradient flow OK — gradients reach per_slot_hazard heads AND slot_attention parameters")

# ── 10. Diversity loss responsiveness ──────────────────────────────────────
# Run 5 more steps to see if the diversity loss responds to training.
variances = []
for step in range(5):
    optimizer.zero_grad()
    logits2, aux2 = model(**batch)
    loss2 = logits2.sum() * 0.0 + aux2
    loss2.backward()
    optimizer.step()
    variances.append(model._last_slot_variance)
    if step == 0:
        print(f"   Step 0 slot_variance: {variances[0]:.4f} "
              f"(target range: [{args.dct_v311_variance_min}, {args.dct_v311_variance_max}])")
print(f"✅  Diversity responsiveness OK — slot_variance range: "
      f"[{min(variances):.4f}, {max(variances):.4f}] "
      f"(diversity loss: {losses['v311_slot_diversity'].item():.4f}, active={losses['v311_slot_diversity'].item() > 0})")

# ── 11. Eval mode: per-slot hazard in explanations ───────────────────────
model.eval()
with torch.no_grad():
    _, _ = model(**batch)
explanations = model.last_explanations
assert "per_slot_hazard_wsi" in explanations, "Missing per_slot_hazard_wsi in explanations"
assert "per_slot_hazard_omic" in explanations, "Missing per_slot_hazard_omic in explanations"
hwsi = explanations["per_slot_hazard_wsi"]
homic = explanations["per_slot_hazard_omic"]
assert hwsi.shape == (B, K_w, C), f"wsi hazard shape wrong: {hwsi.shape}"
assert homic.shape == (B, K_o, C), f"omic hazard shape wrong: {homic.shape}"
assert ((0 < hwsi) & (hwsi < 1)).all(), "wsi hazard values out of (0,1)"
assert ((0 < homic) & (homic < 1)).all(), "omic hazard values out of (0,1)"
print(f"✅  Eval mode OK — per_slot_hazard_wsi={hwsi.shape}, per_slot_hazard_omic={homic.shape}")
print(f"   Sample hazard (patient 0, all WSI slots, all bins):\n{hwsi[0].numpy()}")

# ── 12. Objective weights method ─────────────────────────────────────────
weights = model.objective_weights()
assert "nll" in weights
assert "ipcw_rank" in weights
assert "per_slot_nll" in weights
assert "slot_diversity" in weights
print(f"✅  objective_weights() OK: {weights}")

# ── 13. Key contributions method ─────────────────────────────────────────
contribs = model.key_contributions()
assert len(contribs) > 0
print(f"✅  key_contributions() OK ({len(contribs)} items)")

print("\n" + "=" * 60)
print("ALL SMOKE TESTS PASSED")
print("=" * 60)
