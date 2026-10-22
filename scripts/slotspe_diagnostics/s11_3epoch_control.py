"""Same cosine measurement on the 3-epoch checkpoint, for direct comparison.

s9 measured the 30-epoch checkpoints (mean off-diagonal cosine 0.92 WSI /
0.93 omic).  The earlier 3-epoch pilot reported ~0.9999 with every patient
above the 0.95 threshold.  This script runs the identical measurement on the
3-epoch checkpoint so the difference is attributable to checkpoint age alone
rather than to methodology.

Read-only.  No training, no checkpoint mutation.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data1/DCT-Reg")
sys.path.insert(0, str(REPO))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from survot_rank.research.slotspe_proof.model import build_model, make_args  # noqa: E402
from survot_rank.research.slotspe_proof.patient import _batch  # noqa: E402

sys.path.insert(0, str(REPO / "scripts/slotspe_diagnostics"))
from s9_slot_content_cosine import (  # noqa: E402
    BATCH, DEVICE, SOURCE_ROOT, cosine_matrix, instrumented_forward,
)

RUN_DIR = REPO / "results/slotspe_proof/slotspe_clean_original_blca_run_20261020"
OUT_DIR = REPO / "docs/research/slotspe_slot_content_20261022"
SEED = 20261022


def main():
    from survot_rank.research.slotspe_proof.model import load_source
    manifest = json.loads((RUN_DIR / "prepared/manifest.json").read_text())
    source = load_source(str(SOURCE_ROOT))
    source.MultiHeadSlotAttention.forward = instrumented_forward

    fold_data = manifest["folds"]["0"]
    val_ids = fold_data["val"]
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    args.rna_format = "Pathways"
    model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
    ck = torch.load(RUN_DIR / "pairs/fold_0/seed_3/original/last.pt",
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
            for slots, key in ((s_wsi, "wsi"), (s_omic, "omic")):
                c = cosine_matrix(slots).cpu().numpy()
                rows.append({"bank": key, "cos": c})

    out = {}
    for bank in ("wsi", "omic"):
        c = np.concatenate([r["cos"] for r in rows if r["bank"] == bank], axis=0)
        off = c[:, ~np.eye(8, dtype=bool)]
        out[bank] = {
            "n_patients": int(c.shape[0]),
            "offdiag_mean": float(off.mean()),
            "offdiag_min": float(off.min()),
            "offdiag_max": float(off.max()),
            "frac_above_0.95": float((off > 0.95).mean()),
            "frac_above_0.99": float((off > 0.99).mean()),
            "slot1_slot8_mean": float(c[:, 0, 7].mean()),
            "slot1_slot8_min": float(c[:, 0, 7].min()),
            "slot1_slot8_max": float(c[:, 0, 7].max()),
        }
    out["checkpoint"] = str(RUN_DIR / "pairs/fold_0/seed_3/original/last.pt")
    out["epochs"] = 3
    (OUT_DIR / "slot_content_3epoch.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
