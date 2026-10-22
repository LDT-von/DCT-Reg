"""s15: selected-pair similarity + full 8x8 cosine matrices for SlotSPE original.

Three questions:
  1. Which two slots does the gate keep per modality, and how stable is that choice?
  2. How similar are the two KEPT slots (the "most representative" pair)?
  3. Full 8x8 mean cosine matrices for WSI (pathology) and OMIC (genomic).

Reads only `slot_content.json` (no GPU, no model forward).
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path("/data1/DCT-Reg")
SRC = REPO / "docs/research/slotspe_slot_content_20261022/slot_content.json"
OUT = REPO / "docs/research/slotspe_slot_content_20261022/selected_pair_similarity.json"

data = json.loads(SRC.read_text())

records = []  # per patient
for fold, fdata in data["folds"].items():
    for rec in fdata["per_patient"]:
        records.append((int(fold), rec))

print(f"patients: {len(records)}")


def kept_pair(keep):
    idx = [i for i, k in enumerate(keep) if k > 0.5]
    return idx


report = {"schema": "slotspe_selected_pair_v1", "source": str(SRC), "folds": {}}
summary = {}

for mod in ("wsi", "omic"):
    final_stack, init_stack = [], []
    pair_cos, pair_idx, gate_sums, kept_counts = [], [], [], []
    keep_counter = Counter()

    for fold, rec in records:
        m = rec[mod]
        fc = np.array(m["final_cos"], dtype=np.float64)      # (8,8)
        ic = np.array(m["init_cos"], dtype=np.float64)
        final_stack.append(fc)
        init_stack.append(ic)

        idx = kept_pair(m["keep"])
        kept_counts.append(len(idx))
        keep_counter.update(idx)
        if len(idx) == 2:
            a, b = idx
            pair_cos.append(float(fc[a, b]))
            pair_idx.append((a, b))
        else:
            pair_cos.append(np.nan)
            pair_idx.append(None)
        gate_sums.append(float(np.sum(m["gate"])))

    final_stack = np.stack(final_stack)     # (N,8,8)
    init_stack = np.stack(init_stack)
    pair_cos = np.array(pair_cos)
    gate_sums = np.array(gate_sums)

    mean_final = final_stack.mean(axis=0)
    mean_init = init_stack.mean(axis=0)

    off = ~np.eye(8, dtype=bool)
    off_final = final_stack[:, off]

    # per-slot mean off-diagonal similarity row (how similar is slot i to the rest)
    per_slot_off = np.array([final_stack[:, i, :][:, np.arange(8) != i].mean() for i in range(8)])

    summary[mod] = {
        "n_patients": len(records),
        "kept_count_distribution": dict(Counter(kept_counts)),
        "keep_slot_frequency": {str(k): v / len(records) for k, v in sorted(keep_counter.items())},
        "selected_pair_cosine_mean": float(np.nanmean(pair_cos)),
        "selected_pair_cosine_std": float(np.nanstd(pair_cos)),
        "selected_pair_cosine_median": float(np.nanmedian(pair_cos)),
        "selected_pair_cosine_p05": float(np.nanpercentile(pair_cos, 5)),
        "selected_pair_cosine_p95": float(np.nanpercentile(pair_cos, 95)),
        "selected_pair_cosine_min": float(np.nanmin(pair_cos)),
        "selected_pair_cosine_max": float(np.nanmax(pair_cos)),
        "all_offdiag_cosine_mean": float(off_final.mean()),
        "all_offdiag_cosine_p95": float(np.percentile(off_final, 95)),
        "frac_offdiag_above_0.95": float((off_final > 0.95).mean()),
        "frac_offdiag_above_0.99": float((off_final > 0.99).mean()),
        "mean_final_cosine_matrix": mean_final.tolist(),
        "mean_init_cosine_matrix": mean_init.tolist(),
        "per_slot_mean_offdiag": per_slot_off.tolist(),
        "gate_sum_mean": float(gate_sums.mean()),
        "gate_sum_median": float(np.median(gate_sums)),
        "gate_sum_min": float(gate_sums.min()),
        "gate_sum_max": float(gate_sums.max()),
        "frac_gate_sum_below_0.5": float((gate_sums < 0.5).mean()),
        "frac_gate_sum_below_0.01": float((gate_sums < 0.01).mean()),
    }

    print(f"\n===== {mod.upper()} =====")
    print(f"kept count distribution: {summary[mod]['kept_count_distribution']}")
    print(f"keep slot frequency    : {summary[mod]['keep_slot_frequency']}")
    print(f"selected-pair cosine   : mean={summary[mod]['selected_pair_cosine_mean']:.4f} "
          f"sd={summary[mod]['selected_pair_cosine_std']:.4f} "
          f"p05={summary[mod]['selected_pair_cosine_p05']:.4f} "
          f"p95={summary[mod]['selected_pair_cosine_p95']:.4f}")
    print(f"all off-diagonal       : mean={summary[mod]['all_offdiag_cosine_mean']:.4f} "
          f"frac>0.95={summary[mod]['frac_offdiag_above_0.95']:.4f} "
          f"frac>0.99={summary[mod]['frac_offdiag_above_0.99']:.4f}")
    print(f"gate_sum               : mean={summary[mod]['gate_sum_mean']:.4f} "
          f"median={summary[mod]['gate_sum_median']:.4f} "
          f"frac<0.5={summary[mod]['frac_gate_sum_below_0.5']:.4f} "
          f"frac<0.01={summary[mod]['frac_gate_sum_below_0.01']:.4f}")
    print("mean FINAL 8x8 cosine matrix (rows/cols = slot 1..8):")
    print("      " + "".join(f"{j+1:>9d}" for j in range(8)))
    for i in range(8):
        print(f"slot{i+1} " + "".join(f"{mean_final[i, j]:>9.4f}" for j in range(8)))
    print("mean INIT 8x8 cosine matrix:")
    print("      " + "".join(f"{j+1:>9d}" for j in range(8)))
    for i in range(8):
        print(f"slot{i+1} " + "".join(f"{mean_init[i, j]:>9.4f}" for j in range(8)))

report["summary"] = summary
report["note"] = (
    "SlotSPE original arm (third_party/SlotSPE), 5-fold BLCA seed3, 30-epoch last.pt. "
    "top-k selection uses parallel_topk_st = Gumbel-noised top-k (k = 8 * topk_ratio 0.25 = 2), "
    "so the kept slot pair is RNG-dependent, not a deterministic argmax."
)
OUT.write_text(json.dumps(report, indent=1))
print(f"\nwritten {OUT}")
