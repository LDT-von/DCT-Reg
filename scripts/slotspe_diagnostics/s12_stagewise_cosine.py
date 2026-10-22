"""Where inside the slot attention does the inter-slot similarity appear?

s12 showed the shared `slots_mu` explains only ~1% of the slot norm at 30
epochs, so the shared centre is not the cause of the high inter-slot cosine.
s10 showed the attention is near-uniform (normalised entropy 0.993).

This script records the cosine between the eight slots at every stage of the
attention loop:

    draw      -- after  slots = mu + sigma * noise      (before any iteration)
    updates   -- after  einsum(v, attn)                (attention-weighted
                 average of the value vectors; the per-slot content)
    gru       -- after  the GRU cell update
    mlp       -- after  the residual MLP  (the returned slot)

If the similarity appears at `updates`, the cause is the near-uniform
attention: every slot averages essentially the same bag, so every slot
receives the same update.  If it appears at `gru` or `mlp`, the cause is in
the slot update dynamics instead.

Read-only.  No training, no checkpoint mutation.
"""

from __future__ import annotations

import json
import os
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

RUN = REPO / "results/slotspe_proof/slotspe_full_blca_5fold_20261020"
SRC = REPO / "third_party/SlotSPE"
OUT = REPO / "docs/research/slotspe_slot_content_20261022"
DEV = "cuda:0"
SEED = 20261022
BATCH = 8


def staged_forward(self, inputs, num_slots=None):
    b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    n_s = num_slots if num_slots is not None else self.num_slots
    mu = repeat(self.slots_mu, "1 1 d -> b s d", b=b, s=n_s)
    sigma = repeat(self.slots_logsigma.exp(), "1 1 d -> b s d", b=b, s=n_s)
    slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
    inputs = self.norm_input(inputs)
    k, v = self.to_k(inputs), self.to_v(inputs)
    k, v = map(self.split_heads, (k, v))

    stages = {"draw": slots.detach()}
    for _ in range(self.iters):
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        q = self.split_heads(q)
        dots = einsum("... i d, ... j d -> ... i j", q, k) * self.scale
        attn = dots.softmax(dim=-2)
        attn = F.normalize(attn + self.eps, p=1, dim=-1)
        updates = einsum("... j d, ... i j -> ... i d", v, attn)
        updates = self.merge_heads(updates)
        updates = self.combine_heads(updates)
        stages.setdefault("updates", updates.detach())
        updates, packed_shape = pack([updates], "* d")
        slots_prev, _ = pack([slots_prev], "* d")
        slots = self.gru(updates, slots_prev)
        slots, = unpack(slots, packed_shape, "* d")
        stages.setdefault("gru", slots.detach())
        slots = slots + self.mlp(self.norm_pre_ff(slots))
    stages["mlp"] = slots.detach()
    self._stages = stages
    return slots


def cos_pairs(x):
    xn = F.normalize(x, dim=-1)
    c = torch.einsum("bsd,btd->bst", xn, xn)
    off = ~torch.eye(x.shape[1], dtype=torch.bool, device=x.device)
    return float(c[:, off].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((RUN / "prepared/manifest.json").read_text())
    source = load_source(str(SRC))
    source.MultiHeadSlotAttention.forward = staged_forward

    res = {"schema": "slotspe_slot_stagewise_v1", "seed": SEED, "folds": {}}
    for fold in range(5):
        fd = man["folds"][str(fold)]
        args = make_args(man["omic_sizes"], encoding_dim=man["encoding_dim"])
        args.rna_format = "Pathways"
        m = build_model(str(SRC), args, "original").to(DEV)
        ck = torch.load(RUN / f"pairs/fold_{fold}/seed_3/original/last.pt",
                        map_location=DEV, weights_only=False)
        m.load_state_dict(ck.get("state_dict", ck), strict=True)
        m.eval()

        acc = {}
        for st in range(0, len(fd["val"]), BATCH):
            ids = fd["val"][st:st + BATCH]
            b = _batch(str(RUN), man, fd, ids, DEV, torch)
            with torch.no_grad():
                torch.manual_seed(SEED)
                torch.cuda.manual_seed_all(SEED)
                xw = m.wsi_mlp(b["x_wsi"])
                xo = torch.stack([m.sig_networks[i](b[f"x_omic{i+1}"])
                                  for i in range(m.num_pathways)], dim=1)
                for key, mod, x in (("wsi", m.slot_attention_wsi, xw),
                                    ("omic", m.slot_attention_omic, xo)):
                    mod(x)
                    for stage, vec in mod._stages.items():
                        acc.setdefault(key, {}).setdefault(stage, []).append(cos_pairs(vec))
        res["folds"][str(fold)] = {
            key: {stage: float(np.mean(v)) for stage, v in stages.items()}
            for key, stages in acc.items()}
        print(f"[fold {fold}]", json.dumps(res["folds"][str(fold)]), flush=True)

    # average across folds
    res["mean_over_folds"] = {
        key: {stage: float(np.mean([res["folds"][str(f)][key][stage] for f in range(5)]))
              for stage in res["folds"]["0"][key]}
        for key in ("wsi", "omic")}
    (OUT / "stagewise_cosine.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["mean_over_folds"], indent=1))


if __name__ == "__main__":
    main()
