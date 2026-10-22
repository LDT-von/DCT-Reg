"""Slot content and inter-slot similarity on the trained 30-epoch checkpoints.

Question answered, for BOTH slot banks (8 WSI slots, 8 omic slots):

  * how similar are slot 1 and slot 8 to each other (and every other pair)?
  * how much does each slot actually hold -- i.e. how many distinct patches
    (WSI bank) or pathways (omic bank) does each slot's attention spread over,
    and which ones?

The inter-slot cosine is measured at two stages so the cause is visible:

  init_cos   -- cosine right after the slot draw, before any attention
                iteration.  `slots_mu` has shape (1, 1, dim): ALL EIGHT SLOTS
                SHARE ONE CENTRE VECTOR and differ only by the Gaussian noise
                term, so this stage is high by construction.
  final_cos  -- cosine after the 10 attention iterations, i.e. what the
                trained model actually uses.

Attention is captured from the module's own forward pass (the source forward
is re-implemented verbatim with the assignment matrix retained) and verified
bit-for-bit against the original module output under an identical RNG state.

Read-only.  No training, no checkpoint mutation.
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
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


# --------------------------------------------------------------------------
# instrumented slot attention: the source forward, verbatim, plus the
# assignment matrix (attn) that the original throws away.
# --------------------------------------------------------------------------
def instrumented_forward(self, inputs, num_slots=None):
    b, n, d, device, dtype = *inputs.shape, inputs.device, inputs.dtype
    n_s = num_slots if num_slots is not None else self.num_slots

    mu = repeat(self.slots_mu, "1 1 d -> b s d", b=b, s=n_s)
    sigma = repeat(self.slots_logsigma.exp(), "1 1 d -> b s d", b=b, s=n_s)

    slots = mu + sigma * torch.randn(mu.shape, device=device, dtype=dtype)
    init_slots = slots.clone()

    inputs = self.norm_input(inputs)
    k, v = self.to_k(inputs), self.to_v(inputs)
    k, v = map(self.split_heads, (k, v))

    attn = None
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
        updates, packed_shape = pack([updates], "* d")
        slots_prev, _ = pack([slots_prev], "* d")
        slots = self.gru(updates, slots_prev)
        slots, = unpack(slots, packed_shape, "* d")
        slots = slots + self.mlp(self.norm_pre_ff(slots))

    self._captured = {"attn": attn.detach(), "init_slots": init_slots.detach()}
    return slots


def cosine_matrix(x):
    """x: (B, S, D) -> (B, S, S) cosine similarity."""
    xn = F.normalize(x, dim=-1)
    return torch.einsum("bsd,btd->bst", xn, xn)


def effective_count(w, dim):
    """Participation ratio of a weight vector: 1 / sum(w**2) scaled by n.

    w sums to 1 over `dim` items.  If the weight is uniform, the result is
    `dim`; if it is concentrated on a few items, the result is small.
    """
    return (1.0 / (w ** 2).sum(dim=-1)).clamp_min(1e-12)


def pathway_names():
    """The kept pathway names, in the loader's order (matches manifest sizes)."""
    root = Path("/data1/dataset_csv")
    df = pd.read_csv(root / "signatures/combine_signatures.csv")
    rna = pd.read_csv(root / "raw_rna_data_inter/blca_rna_inter.csv", index_col=0)
    names, sizes = [], []
    for col in df.columns:
        omic = sorted(set(df[col].dropna().unique()).intersection(set(rna.index)))
        if len(omic) == 0:
            continue
        names.append(col)
        sizes.append(len(omic))
    return names, sizes


@contextlib.contextmanager
def fixed_rng(seed):
    py_state, np_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)


def load_model(manifest, fold):
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.rna_format = "Pathways"
    model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
    ckpt_path = RUN_DIR / "pairs" / f"fold_{fold}" / "seed_3" / "original" / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt_path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    names, sizes = pathway_names()
    assert sizes == manifest["omic_sizes"], "pathway alignment broken"

    source = load_source(str(SOURCE_ROOT))
    original_forward = source.MultiHeadSlotAttention.forward
    source.MultiHeadSlotAttention.forward = instrumented_forward

    results = {"schema": "slotspe_slot_content_v1", "seed": SEED, "device": DEVICE,
               "run_dir": str(RUN_DIR), "folds": {}}

    for fold in range(5):
        fold_data = manifest["folds"][str(fold)]
        val_ids = fold_data["val"]
        model, ckpt_path = load_model(manifest, fold)

        per_patient = []
        fidelity = None

        for start in range(0, len(val_ids), BATCH):
            ids = val_ids[start:start + BATCH]
            batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)

            with torch.no_grad(), fixed_rng(SEED):
                x_proj = model.wsi_mlp(batch["x_wsi"])
                x_omic = torch.stack(
                    [model.sig_networks[i](batch[f"x_omic{i + 1}"])
                     for i in range(model.num_pathways)], dim=1)

                s_wsi = model.slot_attention_wsi(x_proj)
                s_omic = model.slot_attention_omic(x_omic)

                cap_w = model.slot_attention_wsi._captured
                cap_o = model.slot_attention_omic._captured

                # fidelity: instrumented forward vs the untouched source module
                # (run after capturing, because it re-runs the module)
                if fidelity is None:
                    with fixed_rng(SEED + 1):
                        ref_w = original_forward(model.slot_attention_wsi, x_proj)
                    with fixed_rng(SEED + 1):
                        got_w = model.slot_attention_wsi(x_proj)
                    fidelity = float((ref_w - got_w).abs().max())

                # attention per slot: average over heads -> (B, S, N)
                aw = cap_w["attn"].mean(dim=1)
                ao = cap_o["attn"].mean(dim=1)

                # decoder gate: which slots are actually used
                _, gate_w, keep_w = model.slot_decoder_wsi(s_wsi)
                _, gate_o, keep_o = model.slot_decoder_omic(s_omic)

                cos_w = cosine_matrix(s_wsi)
                cos_o = cosine_matrix(s_omic)
                init_w = cosine_matrix(cap_w["init_slots"])
                init_o = cosine_matrix(cap_o["init_slots"])

                for b, pid in enumerate(ids):
                    w_w = aw[b]      # (S, N_patches), sums to 1 per slot
                    w_o = ao[b]      # (S, N_pathways)
                    rec = {
                        "patient_id": pid,
                        "wsi": {
                            "final_cos": cos_w[b].cpu().tolist(),
                            "init_cos": init_w[b].cpu().tolist(),
                            "cos_slot1_slot8": float(cos_w[b, 0, 7]),
                            "effective_patches": effective_count(w_w, -1).cpu().tolist(),
                            "top_patches_per_slot": [
                                torch.topk(w_w[s], min(5, w_w.shape[1])).indices.cpu().tolist()
                                for s in range(w_w.shape[0])],
                            "slot_norm": s_wsi[b].norm(dim=-1).cpu().tolist(),
                            "gate": gate_w[b].cpu().tolist(),
                            "keep": keep_w[b].cpu().tolist(),
                        },
                        "omic": {
                            "final_cos": cos_o[b].cpu().tolist(),
                            "init_cos": init_o[b].cpu().tolist(),
                            "cos_slot1_slot8": float(cos_o[b, 0, 7]),
                            "effective_pathways": effective_count(w_o, -1).cpu().tolist(),
                            "top_pathways_per_slot": [
                                [names[j] for j in torch.topk(w_o[s], min(5, w_o.shape[1])).indices.cpu().tolist()]
                                for s in range(w_o.shape[0])],
                            "slot_norm": s_omic[b].norm(dim=-1).cpu().tolist(),
                            "gate": gate_o[b].cpu().tolist(),
                            "keep": keep_o[b].cpu().tolist(),
                        },
                    }
                    per_patient.append(rec)

        results["folds"][str(fold)] = {
            "checkpoint": str(ckpt_path),
            "n_val": len(per_patient),
            "instrumented_forward_max_abs_diff_vs_source": fidelity,
            "per_patient": per_patient,
        }
        print(f"[fold {fold}] {len(per_patient)} patients, fidelity={fidelity:.2e}", flush=True)

    source.MultiHeadSlotAttention.forward = original_forward

    (OUT_DIR / "slot_content.json").write_text(json.dumps(results, indent=1))
    print("written", OUT_DIR / "slot_content.json")


if __name__ == "__main__":
    main()
