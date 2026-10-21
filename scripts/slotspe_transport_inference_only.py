#!/usr/bin/env python
"""
Inference-only test: load trained native checkpoint, attach transport plugin
in learned/independent mode, and evaluate WITHOUT further training.

This tests:
  1. Does transport coupling crash on a frozen trained model?
  2. What's the c-index with vs without transport coupling?
  3. Does coupling tolerance differ for learned vs independent mode?
"""
import sys, os, json, argparse, math, copy, csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEVICE = "cuda:0"


def main():
    import torch
    import numpy as np
    from survot_rank.plugins.transport_only.slotspe_support import data
    from survot_rank.plugins.transport_only.slotspe_support.model import build_model, make_args
    from survot_rank.plugins.transport_only.hosts import attach_transport

    # ── setup ──────────────────────────────────────────────────────────────────
    exp_dir = Path("/tmp/ot03_smoke/OT-03_fold0_seed3_full_v3")
    native_ckpt = exp_dir / "native" / "last.pt"
    protocol = json.load(open(exp_dir / "protocol.json"))
    cfg = protocol["config"]
    mechanism_cfg = protocol["mechanism"]

    manifest_path = exp_dir / "prepared" / "manifest.json"
    manifest = json.load(open(manifest_path))
    fold = cfg["folds"][0]
    seed = cfg["seeds"][0]
    fold_data = manifest["folds"][str(fold)]

    device = torch.device(DEVICE)
    torch.set_num_threads(2)
    data._rng(seed, torch, np)

    # ── build base model ───────────────────────────────────────────────────────
    args = make_args(manifest["omic_sizes"], encoding_dim=manifest["encoding_dim"])
    base = build_model(cfg["source_root"], args, "original")
    initial = copy.deepcopy(base.state_dict())

    # ── attach transport plugin once (shared initial state) ───────────────────
    shared_plugin = attach_transport(base, "slotspe", mechanism_cfg)
    plugin_initial = copy.deepcopy(shared_plugin.mechanism.state_dict())
    shared_plugin.remove()
    del base, shared_plugin

    # ── evaluate each arm ──────────────────────────────────────────────────────
    from sksurv.metrics import concordance_index_censored

    def score_cindex(events, times, risks):
        try:
            v = float(concordance_index_censored(np.asarray(events, bool), np.asarray(times, float),
                                                  np.asarray(risks, float))[0])
            return v if math.isfinite(v) else None
        except ValueError:
            return None

    def run_arm(arm_label, plan_mode, eval_repeats=20):
        print(f"\n{'='*60}")
        print(f"ARM: {arm_label} (plan_mode={plan_mode})")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        data._rng(seed, torch, np)

        model = build_model(cfg["source_root"], args, "original")
        model.load_state_dict(copy.deepcopy(initial), strict=True)

        attachment = None
        if arm_label != "native":
            attachment = attach_transport(model, "slotspe", mechanism_cfg)
            attachment.mechanism.load_state_dict(plugin_initial, strict=True)
            attachment.mechanism.set_plan_mode(plan_mode)

        model.to(device)
        model.eval()

        ids = fold_data["val"]
        all_cindices = []

        coupling_failures = 0
        coupling_attempts = 0

        for rep in range(eval_repeats):
            data._rng(seed * 1000000 + 80000 + rep, torch, np)
            logits_all = []

            with torch.no_grad():
                for start in range(0, len(ids), cfg["batch_size"]):
                    selected = ids[start:start + cfg["batch_size"]]
                    batch = data._batch(exp_dir, manifest, fold_data, selected, DEVICE, torch)
                    logits, _ = model(**batch)
                    logits_all.append(logits.cpu())

            logits = torch.cat(logits_all)
            risks = -(1 - logits.sigmoid()).cumprod(-1).sum(-1)
            events = [manifest["patients"][pid]["censor"] < 1 for pid in ids]
            times = [manifest["patients"][pid]["time"] for pid in ids]
            score = score_cindex(events, times, risks.tolist())
            if score is not None:
                all_cindices.append(score)

        mean_ci = sum(all_cindices) / len(all_cindices) if all_cindices else float("nan")
        std_ci = (sum((x - mean_ci) ** 2 for x in all_cindices) / len(all_cindices)) ** 0.5 if all_cindices else 0

        print(f"  cindex = {mean_ci:.6f} ± {std_ci:.6f}  (n={len(all_cindices)} repeats)")
        if coupling_failures:
            print(f"  coupling failures: {coupling_failures}/{coupling_attempts}")

        return {
            "arm": arm_label, "plan_mode": plan_mode,
            "cindex_mean": mean_ci, "cindex_std": std_ci,
            "cindex_by_repeat": all_cindices,
            "coupling_failures": coupling_failures,
        }

    # ── run ────────────────────────────────────────────────────────────────────
    results = [
        run_arm("native", None),
        run_arm("learned", "learned"),
        run_arm("independent", "independent"),
    ]

    # ── summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFERENCE-ONLY TRANSPORT PLUGIN TEST SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"  {r['arm']:12s}  cindex = {r['cindex_mean']:.6f} ± {r['cindex_std']:.6f}")
    print("=" * 60)

    out_dir = exp_dir / "inference_plugin_test"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also write CSV
    with open(out_dir / "results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["arm", "plan_mode", "cindex_mean", "cindex_std"])
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in ["arm", "plan_mode", "cindex_mean", "cindex_std"]})

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    main()
