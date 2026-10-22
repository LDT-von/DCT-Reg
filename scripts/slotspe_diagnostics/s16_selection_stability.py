"""s16: is the selected slot pair stable, and what does the DETERMINISTIC top-2 look like?

SlotSPE's `parallel_topk_st` adds Gumbel noise before taking the top-k
(k = 8 * topk_ratio 0.25 = 2), so the two "kept" slots are RNG-dependent.

This script measures, on the trained 30-epoch checkpoints:
  A. stability of the kept pair under RNG reseeding (same weights, same patients)
  B. the deterministic (noise-free) top-2 by keep_score, and the cosine between
     that pair -- i.e. the similarity of the two slots the selector *wants*
  C. how often the RNG-kept pair differs from the deterministic pair

Read-only: no training, no checkpoint mutation.
"""
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

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
BASE_SEED = 20261022
REPEATS = 8
BATCH = 8


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_matrix(x):
    xn = torch.nn.functional.normalize(x, dim=-1)
    return torch.einsum("bsd,btd->bst", xn, xn)


def main():
    manifest = json.loads((PREPARED / "manifest.json").read_text())
    results = {"schema": "slotspe_selection_stability_v1", "repeats": REPEATS,
               "base_seed": BASE_SEED, "folds": {}}

    for fold in range(5):
        fold_data = manifest["folds"][str(fold)]
        val_ids = fold_data["val"]

        args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
        args.rna_format = "Pathways"
        model = build_model(str(SOURCE_ROOT), args, "original").to(DEVICE)
        ckpt_path = RUN_DIR / "pairs" / f"fold_{fold}" / "seed_3" / "original" / "last.pt"
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()

        per_patient = []
        for start in range(0, len(val_ids), BATCH):
            ids = val_ids[start:start + BATCH]
            batch = _batch(str(RUN_DIR), manifest, fold_data, ids, DEVICE, torch)

            with torch.no_grad():
                x_proj = model.wsi_mlp(batch["x_wsi"])
                x_omic = torch.stack(
                    [model.sig_networks[i](batch[f"x_omic{i + 1}"])
                     for i in range(model.num_pathways)], dim=1)

                # ---- deterministic keep scores (no Gumbel) ----
                # the slot draw itself is stochastic too, so we need one canonical
                # draw; we use the repeat-0 draw for the deterministic top-2.
                per_repeat = {"wsi": [], "omic": []}
                det = None
                for r in range(REPEATS):
                    set_seed(BASE_SEED + r * 101)
                    s_wsi = model.slot_attention_wsi(x_proj)
                    s_omic = model.slot_attention_omic(x_omic)

                    _, gate_w, keep_w = model.slot_decoder_wsi(s_wsi)
                    _, gate_o, keep_o = model.slot_decoder_omic(s_omic)

                    score_w = model.slot_decoder_wsi.pred_keep_slot(
                        model.slot_decoder_wsi.map(s_wsi)).squeeze(-1)
                    score_o = model.slot_decoder_omic.pred_keep_slot(
                        model.slot_decoder_omic.map(s_omic)).squeeze(-1)

                    cos_w = cosine_matrix(s_wsi)
                    cos_o = cosine_matrix(s_omic)

                    if r == 0:
                        det = {
                            "det_top2_wsi": score_w.topk(2, dim=-1).indices.cpu().tolist(),
                            "det_top2_omic": score_o.topk(2, dim=-1).indices.cpu().tolist(),
                            "score_wsi": score_w.cpu().tolist(),
                            "score_omic": score_o.cpu().tolist(),
                            "cos_wsi": cos_w.cpu(),
                            "cos_omic": cos_o.cpu(),
                        }

                    per_repeat["wsi"].append({
                        "keep": keep_w.cpu().tolist(),
                        "gate": gate_w.cpu().tolist(),
                    })
                    per_repeat["omic"].append({
                        "keep": keep_o.cpu().tolist(),
                        "gate": gate_o.cpu().tolist(),
                    })

            # ---- per-patient records ----
            for b, pid in enumerate(ids):
                rec = {"patient_id": pid}
                for mod, cos_key, det_key in (("wsi", "cos_wsi", "det_top2_wsi"),
                                              ("omic", "cos_omic", "det_top2_omic")):
                    reps = per_repeat[mod]
                    kept_sets = []
                    pair_cos_rng = []
                    for r in range(REPEATS):
                        keep = np.array(reps[r]["keep"][b])
                        idx = [i for i in range(8) if keep[i] > 0.5]
                        kept_sets.append(tuple(idx))
                        if len(idx) == 2:
                            pair_cos_rng.append(float(det[cos_key][b, idx[0], idx[1]]))
                        else:
                            pair_cos_rng.append(float("nan"))

                    # deterministic pair
                    d_idx = det[det_key][b]
                    d_cos = float(det[cos_key][b, d_idx[0], d_idx[1]])

                    uniq = len(set(kept_sets))
                    mode = Counter(kept_sets).most_common(1)[0] if kept_sets else None

                    rec[mod] = {
                        "kept_sets_across_repeats": [list(k) for k in kept_sets],
                        "n_unique_kept_sets": uniq,
                        "modal_kept_set": list(mode[0]) if mode else None,
                        "modal_set_count": mode[1] if mode else 0,
                        "pair_cos_across_repeats": pair_cos_rng,
                        "pair_cos_rng_mean": float(np.nanmean(pair_cos_rng)),
                        "pair_cos_rng_std": float(np.nanstd(pair_cos_rng)),
                        "det_top2": d_idx,
                        "det_pair_cos": d_cos,
                        "det_in_modal_set": bool(set(d_idx).issubset(set(mode[0]))) if mode else False,
                        "score_gap_top2_top3": None,
                    }
                per_patient.append(rec)

        results["folds"][str(fold)] = {"checkpoint": str(ckpt_path),
                                       "n_val": len(per_patient),
                                       "per_patient": per_patient}
        print(f"[fold {fold}] done", flush=True)

    # ---------------- aggregate ----------------
    all_recs = [r for f in results["folds"].values() for r in f["per_patient"]]

    agg = {}
    for mod in ("wsi", "omic"):
        nuniq = [r[mod]["n_unique_kept_sets"] for r in all_recs]
        det_cos = np.array([r[mod]["det_pair_cos"] for r in all_recs])
        rng_mean = np.array([r[mod]["pair_cos_rng_mean"] for r in all_recs])
        rng_std = np.array([r[mod]["pair_cos_rng_std"] for r in all_recs])
        modal_count = np.array([r[mod]["modal_set_count"] for r in all_recs])
        det_in_modal = np.array([r[mod]["det_in_modal_set"] for r in all_recs])

        # which slot pairs does the deterministic selector actually pick?
        det_pairs = Counter(tuple(r[mod]["det_top2"]) for r in all_recs)
        modal_pairs = Counter(tuple(r[mod]["modal_kept_set"]) for r in all_recs)

        agg[mod] = {
            "n_patients": len(all_recs),
            "n_unique_kept_sets_mean": float(np.mean(nuniq)),
            "n_unique_kept_sets_hist": dict(Counter(nuniq)),
            "frac_patients_identical_across_repeats": float(np.mean(np.array(nuniq) == 1)),
            "modal_set_count_mean": float(modal_count.mean()),
            "frac_modal_set_is_deterministic_pair": float(det_in_modal.mean()),
            "det_pair_cosine_mean": float(det_cos.mean()),
            "det_pair_cosine_std": float(det_cos.std()),
            "det_pair_cosine_p05": float(np.percentile(det_cos, 5)),
            "det_pair_cosine_p95": float(np.percentile(det_cos, 95)),
            "rng_pair_cosine_mean": float(rng_mean.mean()),
            "rng_pair_cosine_std_of_patient_mean": float(rng_std.mean()),
            "rng_pair_cosine_p05": float(np.percentile(rng_mean, 5)),
            "rng_pair_cosine_p95": float(np.percentile(rng_mean, 95)),
            "det_pair_histogram": {str(k): v for k, v in det_pairs.most_common(12)},
            "modal_pair_histogram": {str(k): v for k, v in modal_pairs.most_common(12)},
        }
        print(f"\n===== {mod.upper()} =====")
        print(f"kept-set across {REPEATS} RNG repeats: unique count mean={agg[mod]['n_unique_kept_sets_mean']:.2f} "
              f"hist={agg[mod]['n_unique_kept_sets_hist']}")
        print(f"identical across all repeats: {agg[mod]['frac_patients_identical_across_repeats']:.3f}")
        print(f"deterministic top-2 == modal RNG set: {agg[mod]['frac_modal_set_is_deterministic_pair']:.3f}")
        print(f"DET pair cosine : mean={agg[mod]['det_pair_cosine_mean']:.4f} "
              f"sd={agg[mod]['det_pair_cosine_std']:.4f} "
              f"p05={agg[mod]['det_pair_cosine_p05']:.4f} p95={agg[mod]['det_pair_cosine_p95']:.4f}")
        print(f"RNG pair cosine : mean={agg[mod]['rng_pair_cosine_mean']:.4f} "
              f"(within-patient sd {agg[mod]['rng_pair_cosine_std_of_patient_mean']:.4f}) "
              f"p05={agg[mod]['rng_pair_cosine_p05']:.4f} p95={agg[mod]['rng_pair_cosine_p95']:.4f}")
        print(f"top det pairs: {agg[mod]['det_pair_histogram']}")

    results["aggregate"] = agg
    results["note"] = (
        "SlotSPE original arm, 5-fold BLCA seed3, 30-epoch last.pt. "
        "parallel_topk_st = Gumbel-noised top-k (k=2), so the kept pair is RNG-dependent; "
        "'det' = argmax of the raw keep_score without Gumbel noise, using the repeat-0 slot draw."
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "selection_stability.json").write_text(json.dumps(results, indent=1))
    print(f"\nwritten {OUT_DIR / 'selection_stability.json'}")


if __name__ == "__main__":
    main()
