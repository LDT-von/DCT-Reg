"""Why are the eight slots so similar?  Attention entropy and the bag-mean probe.

The 30-epoch measurement (s9) found a paradox:

  * inter-slot cosine is high (mean off-diagonal 0.92 WSI / 0.93 omic)
  * yet the top-5 patch sets of two different slots overlap by only ~1 of 5

So the slots are NOT similar because they attend to the same items.  This
script locates the actual cause:

  entropy        -- is the slot attention near-uniform over the bag?
  max_weight     -- the single largest attention weight per slot
  cos_to_bagmean -- cosine of each slot vector to the (attention-weighted)
                    mean of the inputs, i.e. to the bag average
  slot_vs_bagmean_cos -- cosine of every slot to EVERY OTHER slot, measured
                    in the same space as the bag mean
  uniform_control -- cosine of two independently drawn random subsets of the
                    same size as the attention support

If every slot collapses onto the bag mean, then all slots become mutually
similar AND their attention is near-uniform at the same time, which is exactly
what s9 measured.

Read-only.  No training, no checkpoint mutation.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import pack, repeat, unpack
from torch import einsum

REPO = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from survot_rank.research.slotspe_proof.model import build_model, load_source, make_args  # noqa: E402
from survot_rank.research.slotspe_proof.patient import _batch  # noqa: E402

RUN_DIR = REPO / "results/slotspe_proof/slotspe_full_blca_5fold_20261020"
PREPARED = RUN_DIR / "prepared"
SOURCE_ROOT = REPO / "third_party/SlotSPE"
OUT_DIR = REPO / "docs/research/slotspe_slot_content_20261022"
DEVICE = "cuda:0"
SEED = 20261022
BATCH = 8


def instrumented_forward(self, inputs, num_slots=None):
    b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    n_s = num_slots if num_slots is not None else self.num_slots
    mu = repeat(self.slots_mu, "1 1 d -> b s d", b=b, s=n_s)
    sigma = repeat(self.slots_logsigma.exp(), "1 1 d -> b s d", b=b, s=n_s)
    slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
    inputs = self.norm_input(inputs)
    k, v = self.to_k(inputs), self.to_v(inputs)
    k, v = map(self.split_heads, (k, v))
    for _ in range(self.iters):
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.split_heads(self.to_q(slots))
        dots = einsum("... i d, ... j d -> ... i j", q, k) * self.scale
        attn = dots.softmax(dim=-2)
        attn = F.normalize(attn + self.eps, p=1, dim=-1)
        updates = einsum("... j d, ... i j -> ... i d", v, attn)
        updates = self.combine_heads(self.merge_heads(updates))
        updates, packed_shape = pack([updates], "* d")
        slots_prev, _ = pack([slots_prev], "* d")
        slots = self.gru(updates, slots_prev)
        slots, = unpack(slots, packed_shape, "* d")
        slots = slots + self.mlp(self.norm_pre_ff(slots))
    self._captured = {"attn": attn.detach(), "inputs": inputs.detach()}
    return slots


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    source = load_source(str(SOURCE_ROOT))
    source.MultiHeadSlotAttention.forward = instrumented_forward

    out = {"schema": "slotspe_slot_mechanism_v1", "seed": SEED, "folds": {}}

    for fold in range(5):
        fold_data = manifest["folds"][str(fold)]
        val_ids = fold_data["val"]
        args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
        args.rna_format = "Pathways"
        model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
        ck = torch.load(RUN_DIR / "pairs" / f"fold_{fold}" / "seed_3" / "original" / "last.pt",
                        map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck.get("state_dict", ck), strict=True)
        model.eval()

        rows = []
        for start in range(0, len(val_ids), BATCH):
            ids = val_ids[start:start + BATCH]
            batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)
            with torch.no_grad():
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                x_proj = model.wsi_mlp(batch["x_wsi"])
                x_omic = torch.stack([model.sig_networks[i](batch[f"x_omic{i+1}"])
                                      for i in range(model.num_pathways)], dim=1)
                s_wsi = model.slot_attention_wsi(x_proj)
                s_omic = model.slot_attention_omic(x_omic)

                for mod, slots, key in ((model.slot_attention_wsi, s_wsi, "wsi"),
                                        (model.slot_attention_omic, s_omic, "omic")):
                    attn = mod._captured["attn"].mean(dim=1)      # (B,S,N)
                    inp = mod._captured["inputs"]                  # (B,N,D)
                    for b in range(attn.shape[0]):
                        a = attn[b]                               # (S,N)
                        n = a.shape[1]
                        # normalised entropy (1.0 = uniform)
                        p = a.clamp_min(1e-12)
                        ent = -(p * p.log()).sum(-1) / np.log(n)
                        # attention-weighted mean of the inputs = the bag mean
                        bagmean = torch.einsum("sn,bnd->bsd", a, inp)
                        slots_n = F.normalize(slots[b], dim=-1)     # (S,D)
                        bm_b = F.normalize(bagmean[b], dim=-1)      # (S,D)
                        cos_bm = (slots_n * bm_b).sum(-1)           # (S,)
                        cos_slot_bm = torch.einsum("sd,td->st", slots_n, bm_b)
                        # control: two independent random rows of the same
                        # attention matrix -> how similar are "slots" that are
                        # drawn from the same near-uniform distribution?
                        g = torch.Generator(device=a.device).manual_seed(1234 + b)
                        r1 = torch.rand(2, n, generator=g, device=a.device)
                        r1 = r1 / r1.sum(-1, keepdim=True)
                        cb = torch.einsum("sn,bnd->bsd", r1, inp)
                        cbn = F.normalize(cb, dim=-1)
                        ctrl = (cbn[0] * cbn[1]).sum(-1)
                        rows.append({
                            "bank": key,
                            "entropy": ent.cpu().tolist(),
                            "max_weight": a.max(-1).values.cpu().tolist(),
                            "cos_to_bagmean": cos_bm.cpu().tolist(),
                            "cos_slot_to_bagmean_offdiag": [
                                float(cos_slot_bm[i, j]) for i in range(8) for j in range(8) if i != j],
                            "random_subset_ctrl_cos": float(ctrl.mean()),
                        })
        out["folds"][str(fold)] = rows
        print(f"[fold {fold}] done", flush=True)

    (OUT_DIR / "slot_mechanism.json").write_text(json.dumps(out))
    print("written", OUT_DIR / "slot_mechanism.json")


if __name__ == "__main__":
    main()
