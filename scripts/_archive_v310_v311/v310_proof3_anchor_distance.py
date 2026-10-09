#!/usr/bin/env python3
"""Proof 3: anchor distance ×N intervention audit (BLCA, 5-fold).

Loads v3.10 BLCA ckpts and re-runs redesigned_audit_blca with anchor
L2-distance amplified by factors in {1, 10, 100, 1000}.

Hypothesis:
  - If audit signal (info_gap) **recovers** under large anchor distance
    → root cause is anchor geometry, not direction loss mechanism.
  - If audit signal **stays ~0** even at 1000x → direction loss is broken
    beyond rescue by geometry.

We do NOT retrain.  We only modify model.risk_anchor_costs at audit time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from survot_rank.cli import add_project_paths  # noqa: E402

add_project_paths()

# Reuse redesigned audit's loader + helpers
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from redesign_audit_blca import (  # noqa: E402
    _load_model_and_loader,
    _load_parsed_args,
    _shuffle_slot_dim,
    _interpolate_cost,
    compute_end_to_end,
)
from survot_rank.research.legacy.slotspe_runtime.utils.core_utils import (  # noqa: E402
    _process_data_and_forward,
)

import numpy as np
import torch


def _risk_for_anchor_with_scale(
    model, parsed, val_loader, anchor_idx, device, alpha_list, scale_factor
):
    """Scale the chosen anchor by ``scale_factor`` along its L2 norm, while
    keeping all other risk-level anchors at their original value.

    Returns:
        per_case: list of per-case risk trajectories at each alpha.
        post_l2: L2 distance between scaled anchor and the OTHER anchor
                 (LOW↔HIGH or HIGH↔LOW depending on anchor_idx), measured
                 while the scale is active.

    Effect on LOW/HIGH separation: when called with anchor_idx=LOW at
    scale=10, the LOW anchor's distance from HIGH grows (LOW is pushed
    further from origin while HIGH stays).
    """
    saved = model.risk_anchor_costs[:, anchor_idx].clone()
    model.risk_anchor_costs[:, anchor_idx] = saved * scale_factor

    other_idx = model._HIGH_RISK if anchor_idx == model._LOW_RISK else model._LOW_RISK
    other_saved = model.risk_anchor_costs[:, other_idx].clone()
    # NOTE: we do NOT modify the other anchor.  Reading its L2 against
    # the scaled anchor gives us the actual changed distance.

    try:
        per_case: list[list[float]] = []
        for data in val_loader:
            out, _, _, _ = _process_data_and_forward(
                parsed, model, data, device, test=True
            )
            _, _ = out
            factual_costs = getattr(model, "_last_factual_costs", None)
            if factual_costs is None:
                continue
            rows = model._last_factual_rows
            cols = model._last_factual_cols
            slots_wsi = model._last_slots_wsi
            slots_omic = model._last_slots_omic
            epoch = int(getattr(parsed, "cur_epoch", 0))
            for case_idx in range(factual_costs.size(0)):
                risks: list[float] = []
                for alpha in alpha_list:
                    costs = _interpolate_cost(
                        factual_costs[case_idx:case_idx + 1],
                        model.risk_anchor_costs[:, anchor_idx],
                        alpha,
                        model.risk_anchor_seen[:, anchor_idx],
                    )
                    plan, _ = model._plans_from_cost_tensor(
                        costs,
                        rows[case_idx:case_idx + 1],
                        cols[case_idx:case_idx + 1],
                        epoch,
                    )
                    logits, _ = model._encode_logits_from_plans(
                        slots_wsi[case_idx:case_idx + 1],
                        slots_omic[case_idx:case_idx + 1],
                        plan,
                    )
                    r = float(model._risk(logits).detach().cpu().numpy()[0])
                    risks.append(r)
                per_case.append(risks)
        # Compute the L2 distance between the (now scaled) anchor and the
        # other anchor (still original).  Restore only after reading.
        scaled_now = model.risk_anchor_costs[:, anchor_idx].detach()
        other_now = model.risk_anchor_costs[:, other_idx].detach()
        post_l2 = float(scaled_now.sub(other_now).norm().item())
    finally:
        model.risk_anchor_costs[:, anchor_idx] = saved
        # ``other_saved`` should equal ``other_now`` (we never modified it),
        # but restore explicitly to be safe.
        model.risk_anchor_costs[:, other_idx] = other_saved
    return per_case, post_l2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument(
        "--scale-factors",
        default="1,10,100,1000",
        help="Comma-separated L2-distance multipliers for the chosen anchor",
    )
    parser.add_argument(
        "--output-dir",
        default="results/v310_proof3_anchor_distance",
    )
    parser.add_argument("--epoch", type=int, default=50)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    parsed = _load_parsed_args(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_factors = [float(s) for s in args.scale_factors.split(",")]
    folds = [int(x) for x in args.folds.split(",")]
    epoch_val = int(getattr(args, "epoch", 50))
    parsed.cur_epoch = epoch_val  # mirror redesign_audit_blca.run_fold setup

    alpha_list = (0.0, 1.0)

    summaries = {sf: {} for sf in scale_factors}

    for fold in folds:
        print(f"\n=== fold {fold} ===")
        ckpt = args.checkpoint.replace("s0", f"s{fold}")
        try:
            model, val_loader, _, _, device = _load_model_and_loader(
                type("A", (), {"config": args.config, "checkpoint": ckpt})(),
                parsed,
                fold,
            )
        except Exception as e:
            print(f"  load failed: {e}")
            for sf in scale_factors:
                summaries[sf][fold] = {"error": f"load failed: {e}"}
            continue

        # Measure original L2 distance between LOW and HIGH anchors
        low_a = model.risk_anchor_costs[:, model._LOW_RISK].detach()
        high_a = model.risk_anchor_costs[:, model._HIGH_RISK].detach()
        original_l2 = float(low_a.sub(high_a).norm().item())
        print(f"  original LOW/HIGH L2 distance: {original_l2:.6e}")

        for sf in scale_factors:
            t0 = time.time()
            try:
                real_low, post_l2_low = _risk_for_anchor_with_scale(
                    model, parsed, val_loader, model._LOW_RISK, device, alpha_list, sf
                )
                real_high, post_l2_high = _risk_for_anchor_with_scale(
                    model, parsed, val_loader, model._HIGH_RISK, device, alpha_list, sf
                )
            except Exception as e:
                import traceback
                traceback.print_exc()
                summaries[sf][fold] = {"error": str(e)}
                continue

            metrics_low = compute_end_to_end(real_low, alpha_list)
            metrics_high = compute_end_to_end(real_high, alpha_list)

            # When LOW is scaled, the LOW↔HIGH distance grows (HIGH is original).
            # The two calls give two views of the same LOW↔HIGH L2, since
            # each call leaves the OTHER anchor untouched.
            scaled_l2 = max(post_l2_low, post_l2_high)

            summaries[sf][fold] = {
                "scale_factor": sf,
                "scaled_l2_distance": scaled_l2,
                "ratio_l2": scaled_l2 / max(original_l2, 1e-30),
                "real_low": metrics_low,
                "real_high": metrics_high,
                "elapsed_sec": round(time.time() - t0, 1),
            }
            print(
                f"  scale={sf:>6.1f} | new_l2={scaled_l2:.4e} "
                f"(x{scaled_l2/original_l2:.2f}) | "
                f"high_rate_up={metrics_high['rate_up']:.3f} "
                f"low_rate_down={metrics_low['rate_down']:.3f}"
            )

        # Free model
        del model
        torch.cuda.empty_cache()

    # Save per-scale json
    for sf in scale_factors:
        json_path = output_dir / f"scale_{sf:g}.json"
        with open(json_path, "w") as f:
            json.dump(summaries[sf], f, indent=2)

    # Aggregate per-scale 5-fold mean
    agg = {}
    for sf in scale_factors:
        h_up = [
            summaries[sf][f]["real_high"]["rate_up"]
            for f in folds
            if "real_high" in summaries[sf].get(f, {})
        ]
        l_dn = [
            summaries[sf][f]["real_low"]["rate_down"]
            for f in folds
            if "real_low" in summaries[sf].get(f, {})
        ]
        l2_dist = [
            summaries[sf][f]["scaled_l2_distance"]
            for f in folds
            if "scaled_l2_distance" in summaries[sf].get(f, {})
        ]
        if h_up and l_dn:
            agg[str(sf)] = {
                "scale_factor": sf,
                "l2_distance_mean": float(np.mean(l2_dist)),
                "high_rate_up_mean": float(np.mean(h_up)),
                "high_rate_up_std": float(np.std(h_up)),
                "low_rate_down_mean": float(np.mean(l_dn)),
                "low_rate_down_std": float(np.std(l_dn)),
            }
    with open(output_dir / "summary_by_scale.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Markdown
    lines = [
        "# v3.10 Proof 3: Anchor Distance × N Intervention (BLCA, 5-fold)\n",
        "**核心问题**:v3.10 的 redesigned audit 测出来 info_gap ≈ 0,这是**anchor 几何问题**,还是 **direction loss 机制本身的问题**?\n",
        "\n",
        "**实验**:加载 v3.10 BLCA 5 fold ckpt,**不重训**,只在 audit 时把 LOW/HIGH anchor 沿原始方向放大 scale 倍,跑 redesigned audit,看 info_gap 随 scale 的变化。\n",
        "\n",
        "**两种解读**:\n",
        "- **如果** `high_rate_up` 或 `low_rate_down` 随 scale 单调上升(信号回来了)→ anchor 距离是 root cause,direction loss 设计是对的,只是 anchor 几何太挤。\n",
        "- **如果** 始终不变 → direction loss 机制本身有问题(配合 Proof 2 的\"梯度衰减 30 万倍\"结论)。\n",
        "\n",
        "## 5-fold mean by scale\n",
        "\n",
        "| Scale | L2 distance | high_rate_up | low_rate_down |",
        "|---|---|---|---|",
    ]
    for sf in scale_factors:
        if str(sf) in agg:
            a = agg[str(sf)]
            lines.append(
                f"| {sf:g} | {a['l2_distance_mean']:.4e} | "
                f"{a['high_rate_up_mean']:.3f} ± {a['high_rate_up_std']:.3f} | "
                f"{a['low_rate_down_mean']:.3f} ± {a['low_rate_down_std']:.3f} |"
            )

    lines.extend([
        "\n## Per-fold\n",
        "\n",
        "| Fold |",
    ])
    header_row = "| Fold |"
    for sf in scale_factors:
        header_row += f" s={sf:g} high_up | s={sf:g} low_down |"
    lines[-1] = header_row
    lines.append("|" + "---|" * (1 + 2 * len(scale_factors)))
    for fold in folds:
        row = f"| {fold} |"
        for sf in scale_factors:
            entry = summaries[sf].get(fold, {})
            if "real_high" in entry:
                row += f" {entry['real_high']['rate_up']:.3f} | {entry['real_low']['rate_down']:.3f} |"
            else:
                row += f" err | err |"
        lines.append(row)

    lines.extend([
        "\n## 解读\n",
        "\n",
        "对比 scale=1(原始)与 scale=100/1000:\n",
        "- 若 high_rate_up / low_rate_down 在大 scale 下**显著上升**(如高 20pp+)\n",
        "  → 证明 anchor 几何是 audit 信号弱的**根因**。\n",
        "- 若几乎不变 → 配合 Proof 2 (direction 梯度衰减 30万倍)\n",
        "  → **direction loss 机制本身失败**,anchor 几何不是问题。\n",
    ])
    md_path = output_dir / "REPORT.md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nsaved {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
